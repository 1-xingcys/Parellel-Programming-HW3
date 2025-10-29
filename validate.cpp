#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip> // 用於 std::setprecision

#include "lodepng.h" // 假設 lodepng.h 在 include path 或同一目錄

/**
 * @brief 讀取 PNG 檔案並解碼到 std::vector<unsigned char>
 * * @param image_data (輸出) 儲存 RGBA 像素資料的 vector
 * @param width (輸出) 圖片寬度
 * @param height (輸出) 圖片高度
 * @param filename (輸入) 要讀取的 PNG 檔案路徑
 * @return true 如果成功, false 如果失敗
 */
bool decodePNG(std::vector<unsigned char>& image_data, unsigned& width, unsigned& height, const std::string& filename) {
    // lodepng::decode 會自動將圖片轉為 32-bit RGBA 格式
    unsigned error = lodepng::decode(image_data, width, height, filename);

    // 如果有錯誤
    if (error) {
        std::cerr << "LodePNG 解碼錯誤 " << error << ": " << lodepng_error_text(error) << " 於檔案: " << filename << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    // 檢查是否有足夠的參數 (程式名稱 + 兩個圖檔)
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <image1.png> <image2.png>" << std::endl;
        return 1;
    }

    std::string filename1 = argv[1];
    std::string filename2 = argv[2];

    std::vector<unsigned char> image1_data;
    unsigned width1, height1;
    std::vector<unsigned char> image2_data;
    unsigned width2, height2;

    // --- 1. 載入第一張圖片 ---
    std::cout << "正在載入: " << filename1 << "..." << std::endl;
    if (!decodePNG(image1_data, width1, height1, filename1)) {
        return 1;
    }

    // --- 2. 載入第二張圖片 ---
    std::cout << "正在載入: " << filename2 << "..." << std::endl;
    if (!decodePNG(image2_data, width2, height2, filename2)) {
        return 1;
    }

    // --- 3. 檢查尺寸 ---
    if (width1 != width2 || height1 != height2) {
        std::cout << "錯誤：圖片尺寸不同。" << std::endl;
        std::cout << "  " << filename1 << ": " << width1 << "x" << height1 << std::endl;
        std::cout << "  " << filename2 << ": " << width2 << "x" << height2 << std::endl;
        std::cout << "\nPixel 相符程度: 0.0000%" << std::endl;
        return 0;
    }

    std::cout << "圖片尺寸相同: " << width1 << "x" << height1 << std::endl;

    // --- 4. 逐一 pixel 比較 ---
    unsigned long long matching_pixels = 0;
    unsigned long long total_pixels = width1 * height1;

    // 圖片資料是 1D 陣列 [R1, G1, B1, A1, R2, G2, B2, A2, ...]
    for (unsigned y = 0; y < height1; ++y) {
        for (unsigned x = 0; x < width1; ++x) {
            
            // 計算 1D 索引 (y * width + x) * 4 (因為 RGBA 四個 channel)
            size_t index = (y * width1 + x) * 4;

            // 讀取 圖 1 的 RGBA
            unsigned char r1 = image1_data[index + 0];
            unsigned char g1 = image1_data[index + 1];
            unsigned char b1 = image1_data[index + 2];
            unsigned char a1 = image1_data[index + 3];

            // 讀取 圖 2 的 RGBA
            unsigned char r2 = image2_data[index + 0];
            unsigned char g2 = image2_data[index + 1];
            unsigned char b2 = image2_data[index + 2];
            unsigned char a2 = image2_data[index + 3];

            // 比較：四個 channel 必須完全一樣
            if (r1 == r2 && g1 == g2 && b1 == b2 && a1 == a2) {
                matching_pixels++;
            }
            
            // (可選) 如果需要找出不相符的 pixel 在哪裡，可以在此處加入 else：
            // else {
            //    std::cout << "Pixel 不符 @ (" << x << ", " << y << ")" << std::endl;
            // }
        }
    }

    // --- 5. 計算並顯示結果 ---
    double match_percentage = 0.0;
    if (total_pixels > 0) {
        match_percentage = static_cast<double>(matching_pixels) / static_cast<double>(total_pixels) * 100.0;
    }

    std::cout << "\n--- 比較結果 ---" << std::endl;
    std::cout << std::fixed << std::setprecision(4); // 顯示到小數點後 4 位
    std::cout << "Pixel 相符程度: " << match_percentage << "%" << "(" << matching_pixels << "/" << total_pixels << ")" << std::endl;

    return 0;
}
