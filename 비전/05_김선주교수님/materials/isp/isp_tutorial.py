import rawpy
import numpy as np
import imageio
import cv2
import os

def camera_isp_tutorial(raw_image_path, output_dir):
    """
    카메라 ISP 파이프라인을 단계별로 실행하고 중간 결과물을 저장하는 튜토리얼 함수입니다.

    Args:
        raw_image_path (str): 입력 RAW 이미지 파일 경로 (.dng)
        output_dir (str): 결과 이미지를 저장할 디렉토리 경로
    """
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"'{output_dir}' 디렉토리를 생성했습니다.")

    # --- 1. RAW 파일 읽기 ---
    print("--- 단계 1: RAW 파일 읽기 시작 ---")
    try:
        raw = rawpy.imread(raw_image_path)
        print("RAW 파일 읽기 완료.")
    except rawpy.LibRawNoBadPixelPathError:
         # 이 오류는 DNG 파일에 bad pixel map이 없을 때 발생할 수 있습니다.
         # 이 경우 `no_auto_scale` 옵션을 사용해 무시하고 진행할 수 있습니다.
        raw = rawpy.imread(raw_image_path, params=rawpy.Params(no_auto_scale=True))
        print("경고를 무시하고 RAW 파일을 읽었습니다.")
    except Exception as e:
        print(f"RAW 파일 읽기 오류: {e}")
        return

    # --- 2. 블랙 레벨 보정 (Linearization) ---
    # 센서의 노이즈 플로어인 블랙 레벨을 빼주어 빛에 대한 반응을 선형적으로 만듭니다.
    print("\n--- 단계 2: 블랙 레벨 보정 시작 ---")
    black_level = np.array(raw.black_level_per_channel, dtype=np.float32)
    white_level = float(raw.white_level)

    # raw_image_visible는 이미 블랙 레벨이 어느 정도 보정된 데이터를 제공합니다.
    # 여기서는 좀 더 명시적인 제어를 위해 직접 계산합니다.
    bayer_pattern = raw.raw_image_visible.astype(np.float32)
    print("bayer_pattern.shape:", bayer_pattern.shape)
    
    # 블랙 레벨 빼기
    bayer_pattern -= black_level.mean() # 채널별 black level이 다를 수 있어 평균 사용
    
    # 0 이하 값 클리핑
    bayer_pattern = np.maximum(bayer_pattern, 0)

    # 화이트 레벨로 정규화 (0.0 ~ 1.0 범위)
    normalized_bayer = bayer_pattern / (white_level - black_level.mean())
    normalized_bayer = np.clip(normalized_bayer, 0, 1)

    # 시각화를 위해 8비트로 변환하여 저장
    bayer_8bit = (normalized_bayer * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(output_dir, "step1_normalized_bayer.png"), bayer_8bit)
    print("블랙 레벨 보정 및 정규화 완료. 'step1_normalized_bayer.png' 저장.")

    # --- 3. 디모자이킹 (Demosaicing / Debayering) ---
    # 베이어 패턴의 단일 색상 정보를 보간하여 각 픽셀이 R, G, B 값을 모두 갖도록 합니다.
    print("\n--- 단계 3: 디모자이킹 시작 ---")
    # RAW 데이터의 차원과 채널 수를 확인하여 처리
    if normalized_bayer.ndim == 3 and normalized_bayer.shape[2] == 4:
        # CASE 1: 4채널 이미지 (RGB + 사용되지 않는 Alpha 채널)
        print("4-채널 이미지(RGBA)가 감지되었습니다. 마지막 채널을 무시하고 RGB 데이터만 사용합니다.")
        
        # NumPy 슬라이싱으로 처음 3개 채널(R, G, B)만 선택
        linear_rgb = normalized_bayer[:, :, :3]

    elif normalized_bayer.ndim == 3 and normalized_bayer.shape[2] == 3:
        # CASE 2: 3채널 Linear DNG (RGB)
        print("3-채널 RGB 이미지가 감지되었습니다. 디모자이킹을 건너뜁니다.")
        linear_rgb = normalized_bayer

    else:
        print("1채널 Bayer 패턴 이미지입니다. 디모자이킹을 수행합니다.")
        # rawpy가 제공하는 베이어 패턴 정보를 OpenCV 플래그로 변환
        # raw.raw_pattern은 0=R, 1=G, 2=B, 3=G' 를 나타내는 2x2 numpy 배열
        pattern_str = "".join(map(chr, raw.color_desc))
        print(pattern_str)
        
        # OpenCV는 BGGR, GBRG, GRBG, RGGB 순서를 사용합니다.
        if pattern_str[:2] == 'RG':
            demosaic_flag = cv2.COLOR_BAYER_RG2BGR
        elif pattern_str[:2] == 'BG':
            demosaic_flag = cv2.COLOR_BAYER_BG2BGR
        elif pattern_str[:2] == 'GR':
            demosaic_flag = cv2.COLOR_BAYER_GR2BGR
        elif pattern_str[:2] == 'GB':
            demosaic_flag = cv2.COLOR_BAYER_GB2BGR
        else:
            raise ValueError(f"지원하지 않는 베이어 패턴: {pattern_str}")
            
        # 정규화된 베이어 이미지를 16비트로 변환하여 디모자이킹 (정밀도 유지)
        bayer_16bit = (normalized_bayer * 65535).astype(np.uint16)
        linear_rgb = cv2.cvtColor(bayer_16bit, demosaic_flag)
        linear_rgb = linear_rgb.astype(np.float32) / 65535.0
    
    # 다시 0.0 ~ 1.0 범위의 float으로 변환
    linear_rgb = linear_rgb.astype(np.float32)
    
    imageio.imwrite(os.path.join(output_dir, "step2_demosaiced_linear_rgb.png"), (np.clip(linear_rgb, 0, 1) * 255).astype(np.uint8))
    print("디모자이킹 완료. 'step2_demosaiced_linear_rgb.png' 저장. (이미지가 어둡게 보일 수 있음)")

    # --- 4. 화이트 밸런스 (White Balance) ---
    # 촬영 환경의 광원 색온도를 보정하여 흰색이 흰색으로 보이게 만듭니다.
    print("\n--- 단계 4: 화이트 밸런스 적용 시작 ---")
    # DNG에 기록된 카메라의 화이트 밸런스 값 사용 [B, G, R, G]
    wb_multipliers = np.array(raw.camera_whitebalance, dtype=np.float32)
    # R과 B 채널에 대한 보정값 (G 채널을 1로 기준)
    wb_coeffs = wb_multipliers[:3] / wb_multipliers[1]
    
    # R, G, B 채널에 각각 보정 계수 곱하기
    wb_rgb = np.copy(linear_rgb)
    wb_rgb[:, :, 0] *= wb_coeffs[0]  # B
    wb_rgb[:, :, 1] *= wb_coeffs[1]  # G (1.0)
    wb_rgb[:, :, 2] *= wb_coeffs[2]  # R

    # 클리핑
    wb_rgb = np.clip(wb_rgb, 0, 1)

    imageio.imwrite(os.path.join(output_dir, "step3_white_balanced.png"), (wb_rgb * 255).astype(np.uint8))
    print("화이트 밸런스 적용 완료. 'step3_white_balanced.png' 저장.")

    # --- 5. 색 공간 변환 (Color Space Conversion) ---
    # 카메라 센서의 고유 색 공간에서 표준 sRGB 색 공간으로 변환합니다.
    print("\n--- 단계 5: 색 공간 변환 시작 ---")
    
    # raw.color_matrix 속성은 카메라의 RGB에서 sRGB로 변환하는
    # 색 보정 행렬(CCM)을 포함하고 있습니다.
    # 이 행렬의 3x3 부분만 사용합니다.
    ccm = raw.color_matrix[:3, :3].astype(np.float32)

    # 화이트 밸런스가 적용된 이미지에 색 보정 행렬을 적용합니다.
    h, w, c = wb_rgb.shape
    
    # 효율적인 계산을 위해 이미지 배열을 (높이*너비, 채널) 형태로 변경
    wb_rgb_reshaped = wb_rgb.reshape(-1, c)
    
    # 행렬 곱셈을 통해 색 공간 변환 수행
    srgb_linear_reshaped = np.dot(wb_rgb_reshaped, ccm.T)
    
    # 변환된 이미지를 다시 원래의 (높이, 너비, 채널) 형태로 복원
    srgb_linear = srgb_linear_reshaped.reshape(h, w, c)
    
    # 0.0 ~ 1.0 범위 밖의 값을 잘라냅니다.
    srgb_linear = np.clip(srgb_linear, 0, 1)

    imageio.imwrite(os.path.join(output_dir, "step4_srgb_linear.png"), (srgb_linear * 255).astype(np.uint8))
    print("색 공간 변환 완료. 'step4_srgb_linear.png' 저장.")
    
    # --- 6. 감마 보정 (Gamma Correction) ---
    # 사람의 시각이 밝기를 비선형적으로 인지하는 것에 맞춰 이미지의 밝기를 조절합니다.
    print("\n--- 단계 6: 감마 보정 시작 ---")
    
    # sRGB 감마 커브 적용 (표준 EOTF의 간략화 버전)
    gamma = 2.2
    srgb_gamma_corrected = np.power(srgb_linear, 1/gamma)
    
    # 클리핑
    srgb_gamma_corrected = np.clip(srgb_gamma_corrected, 0, 1)
    
    # 8비트로 최종 변환
    srgb_image = (srgb_gamma_corrected * 255).astype(np.uint8)

    imageio.imwrite(os.path.join(output_dir, "step5_srgb_image.png"), srgb_image)
    print("감마 보정 완료. 'step5_srgb_image.png' 저장.")

    print("\n--- 단계 7: 사진 회전 시작 ---")
    rotation_angle = raw.sizes.flip
    rotated_image = srgb_image
    if rotation_angle == 6: # 90도 시계방향 회전
        rotated_image = np.rot90(srgb_image, k=3)
    elif rotation_angle == 3: # 180도 회전
        rotated_image = np.rot90(srgb_image, k=2)
    imageio.imwrite(os.path.join(output_dir, "step6_rotated_image.png"), rotated_image)
    print("사진 회전 완료. 최종 결과물. 'step6_rotated_image.png' 저장.")

    # --- (비교용) rawpy의 postprocess() 함수 사용 ---
    # 위 모든 과정을 한번에 처리하는 고수준 함수
    print("\n--- 비교: rawpy의 postprocess() 함수로 한번에 처리 ---")
    processed_image = raw.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB, no_auto_bright=False, gamma=(2.2, 1))
    imageio.imwrite(os.path.join(output_dir, "compare_rawpy_auto.png"), processed_image)
    print("rawpy 자동 처리 완료. 'compare_rawpy_auto.png' 저장.")

    raw.close()
    print("\n🎉 모든 ISP 과정이 완료되었습니다.")


if __name__ == '__main__':
    # 여기에 본인의 DNG 파일 경로를 입력하세요.
    input_dng_file = 'iphone_15pro.dng'
    
    # 결과물이 저장될 폴더 이름을 지정합니다.
    output_folder = 'isp_output'

    if not os.path.exists(input_dng_file):
        print(f"오류: 입력 파일 '{input_dng_file}'을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    else:
        camera_isp_tutorial(input_dng_file, output_folder)