#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// CUDA 執行時標頭檔
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 用於 device 端的向量類型 (如 double3, double2)
#include <vector_types.h>

// LodePNG 用於儲存影像
#include <lodepng.h>

// 移除 GLM 包含

#define pi 3.14159265358979323846

// 原始 C++ 程式碼中的全域變數，現在移入此結構體
struct RenderParams {
    double power;
    int md_iter;
    int ray_step;
    int shadow_step;
    double step_limiter;
    double ray_multiplier;
    double bailout;
    double eps;
    double FOV;
    double far_plane;
    double3 camera_pos;
    double3 target_pos;
    double2 iResolution;
    int AA;
};

// 在 GPU 的常數記憶體中宣告一個 RenderParams 實例
// __constant__ 記憶體會被快取並高效廣播
__constant__ RenderParams d_params;

// 原始 CPU 程式碼中的變數 (用於 main 函式初始化)
int AA = 3;
double power = 8.0;
double md_iter = 24;
double ray_step = 10000;
double shadow_step = 1500;
double step_limiter = 0.2;
double ray_multiplier = 0.1;
double bailout = 2.0;
double eps = 0.0005;
double FOV = 1.5;
double far_plane = 100.;


// CUDA 錯誤檢查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


//==============================================================================
// Host (CPU) 程式碼
//==============================================================================

// save raw_image to PNG file (與原始碼相同)
void write_png(const char* filename, unsigned char* raw_image, unsigned int width, unsigned int height) {
    unsigned error = lodepng_encode32_file(filename, raw_image, width, height);
    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));
}

//==============================================================================
// Device (GPU) 程式碼
//==============================================================================

// mandelbulb distance function (DE)
// __device__ 表示此函式在 GPU 上執行
__device__ double md(double3 p, double& trap) {
    double3 v = p;
    double dr = 1.;
    double r = length(v); // glm::length -> length (CUDA built-in)
    trap = r;

    // 使用常數記憶體中的參數
    for (int i = 0; i < d_params.md_iter; ++i) {
        // glm::atan(v.y, v.x) -> atan2(v.y, v.x)
        double theta = atan2(v.y, v.x) * d_params.power;
        // glm::asin(v.z / r) -> asin(v.z / r)
        double phi = asin(v.z / r) * d_params.power;
        
        // glm::pow -> pow
        dr = d_params.power * pow(r, d_params.power - 1.) * dr + 1.;
        
        double r_pow = pow(r, d_params.power);
        // make_double3 for vec3(...)
        double3 v_new = make_double3(cos(theta) * cos(phi), 
                                    cos(phi) * sin(theta), 
                                    -sin(phi));
        v = p + r_pow * v_new;

        // glm::min -> fmin (for doubles)
        trap = fmin(trap, r);
        r = length(v);
        if (r > d_params.bailout) break;
    }
    return 0.5 * log(r) * r / dr;
}

// scene mapping
__device__ double map(double3 p, double& trap, int& ID) {
    // vec2 rt = vec2(cos(pi / 2.), sin(pi / 2.)); // (0, 1)
    // mat3(...) * p;
    // 原始碼的矩陣乘法是繞 X 軸旋轉 90 度
    // (x, y, z) -> (x, z, -y)
    double3 rp = make_double3(p.x, p.z, -p.y);
    ID = 1;
    return md(rp, trap);
}

// dummy function overload
__device__ double map(double3 p) {
    double dmy;
    int dmy2;
    return map(p, dmy, dmy2);
}

// simple palette function (borrowed from Inigo Quilez)
__device__ double3 pal(double t, double3 a, double3 b, double3 c, double3 d) {
    // glm::cos -> cos. CUDA Math API 會處理 double
    double cos_val_r = cos(2. * pi * (c.x * t + d.x));
    double cos_val_g = cos(2. * pi * (c.y * t + d.y));
    double cos_val_b = cos(2. * pi * (c.z * t + d.z));
    
    // (a, b, c, d 都是 double3)
    return a + b * make_double3(cos_val_r, cos_val_g, cos_val_b);
}

// second march: cast shadow
__device__ double softshadow(double3 ro, double3 rd, double k) {
    double res = 1.0;
    double t = 0.;
    for (int i = 0; i < d_params.shadow_step; ++i) {
        double h = map(ro + rd * t);
        res = fmin(res, k * h / t); // glm::min -> fmin
        if (res < 0.02) return 0.02;
        // glm::clamp -> clamp (CUDA built-in)
        t += clamp(h, .001, d_params.step_limiter);
    }
    return clamp(res, .02, 1.);
}

// use gradient to calc surface normal
__device__ double3 calcNor(double3 p) {
    // vec2 e = vec2(eps, 0.); -> double2 e
    double2 e = make_double2(d_params.eps, 0.0);
    
    // e.xyy() -> (eps, 0, 0)
    double3 e_xyy = make_double3(e.x, e.y, e.y);
    // e.yxy() -> (0, eps, 0)
    double3 e_yxy = make_double3(e.y, e.x, e.y);
    // e.yyx() -> (0, 0, eps)
    double3 e_yyx = make_double3(e.y, e.y, e.x);

    return normalize(make_double3( // glm::normalize -> normalize
        map(p + e_xyy) - map(p - e_xyy),
        map(p + e_yxy) - map(p - e_yxy),
        map(p + e_yyx) - map(p - e_yyx)
    ));
}

// first march: find object's surface
__device__ double trace(double3 ro, double3 rd, double& trap, int& ID) {
    double t = 0;
    double len = 0;

    for (int i = 0; i < d_params.ray_step; ++i) {
        len = map(ro + rd * t, trap, ID);
        // glm::abs -> fabs (for double)
        if (fabs(len) < d_params.eps || t > d_params.far_plane) break;
        t += len * d_params.ray_multiplier;
    }
    return t < d_params.far_plane ? t : -1.;
}


//==============================================================================
// CUDA Kernel (Global Function)
//==============================================================================

// __global__ 表示這是一個 Kernel，可由 CPU 啟動並在 GPU 上執行
__global__ void render_kernel(unsigned char* d_image, int width, int height) {
    
    // 計算全域執行緒 ID (對應到像素座標)
    // blockIdx, blockDim, threadIdx 是 CUDA 內建變數
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 像素 x 座標 (width)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 像素 y 座標 (height)

    // 邊界檢查：防止執行緒處理超出影像範圍的像素
    if (j >= width || i >= height) {
        return;
    }

    // --- 程式碼主體：從 CPU main() 迴圈中複製而來 ---
    
    // vec4 fcol(0.); -> 使用 double
    double fcol_r = 0.0;
    double fcol_g = 0.0;
    double fcol_b = 0.0;

    // anti aliasing (使用常數記憶體中的 d_params.AA)
    for (int m = 0; m < d_params.AA; ++m) {
        for (int n = 0; n < d_params.AA; ++n) {
            
            // vec2 p = vec2(j, i) + ...
            double2 p = make_double2((double)j, (double)i) + 
                       make_double2((double)m, (double)n) / (double)d_params.AA;

            // vec2 uv = ... (使用 d_params.iResolution)
            double2 uv = (-d_params.iResolution + 2. * p) / d_params.iResolution.y;
            uv.y *= -1;

            // create camera
            // (使用 d_params.camera_pos, target_pos, FOV)
            double3 ro = d_params.camera_pos;
            double3 ta = d_params.target_pos;
            double3 cf = normalize(ta - ro); // glm::normalize -> normalize
            // glm::cross -> cross
            double3 cs = normalize(cross(cf, make_double3(0., 1., 0.))); 
            double3 cu = normalize(cross(cs, cf));
            double3 rd = normalize(uv.x * cs + uv.y * cu + d_params.FOV * cf);

            // marching
            double trap;
            int objID;
            double d = trace(ro, rd, trap, objID);

            // lighting
            // vec3 col(0.);
            double3 col = make_double3(0.0, 0.0, 0.0);
            double3 sd = normalize(d_params.camera_pos); // Sun direction
            double3 sc = make_double3(1., .9, .717);     // Light color

            // coloring
            if (d < 0.) { // miss (hit sky)
                col = make_double3(0.0, 0.0, 0.0); // sky color (black)
            } else {
                double3 pos = ro + rd * d;
                double3 nr = calcNor(pos);
                double3 hal = normalize(sd - rd); // blinn-phong

                // vec3(.5) -> make_double3(.5, .5, .5)
                col = pal(trap - .4, make_double3(.5, .5, .5), make_double3(.5, .5, .5), 
                          make_double3(1., 1., 1.), make_double3(.0, .1, .2));
                double3 ambc = make_double3(0.3, 0.3, 0.3);
                double gloss = 32.;

                // simple blinn phong
                // glm::clamp -> clamp
                double amb = (0.7 + 0.3 * nr.y) * (0.2 + 0.8 * clamp(0.05 * log(trap), 0.0, 1.0));
                double sdw = softshadow(pos + 0.001 * nr, sd, 16.);
                // glm::dot -> dot
                double dif = clamp(dot(sd, nr), 0., 1.) * sdw;
                double spe = pow(clamp(dot(nr, hal), 0., 1.), gloss) * dif;

                double3 lin = make_double3(0.0, 0.0, 0.0);
                lin += ambc * (.05 + .95 * amb);
                lin += sc * dif * 0.8;
                col *= lin;

                // glm::pow(col, vec3(...)) -> pow(col, make_double3(...))
                // (注意: pow(double3, double3) 在 CUDA 中可能不直接支援，
                // 但 pow(double3, double) 可以。這裡原始碼是 pow(vec, vec)。)
                // (更正：CUDA Math API 沒有 pow(double3, double3)。
                // 原始碼的 glm::pow(vec, vec) 是逐元素 pow。我們手動實現。)
                col = make_double3(pow(col.x, 0.7), pow(col.y, 0.9), pow(col.z, 1.0));
                col += spe * 0.8;
            }

            // Gamma correction (glm::pow(col, vec3(.4545)))
            col = clamp(pow(col, make_double3(0.4545, 0.4545, 0.4545)), 0.0, 1.0);
            
            // fcol += vec4(col, 1.);
            fcol_r += col.x; // .x for double3
            fcol_g += col.y; // .y for double3
            fcol_b += col.z; // .z for double3
        }
    }
    // --- 迴圈結束 ---

    // fcol /= (double)(AA * AA);
    fcol_r /= (double)(d_params.AA * d_params.AA);
    fcol_g /= (double)(d_params.AA * d_params.AA);
    fcol_b /= (double)(d_params.AA * d_params.AA);
    
    // fcol *= 255.0;
    fcol_r *= 255.0;
    fcol_g *= 255.0;
    fcol_b *= 255.0;

    // 計算 1D 索引並寫入全域記憶體
    int index = (i * width + j) * 4;
    d_image[index + 0] = (unsigned char)fcol_r;
    d_image[index + 1] = (unsigned char)fcol_g;
    d_image[index + 2] = (unsigned char)fcol_b;
    d_image[index + 3] = 255;
}


//==============================================================================
// Host (CPU) Main Function
//==============================================================================

int main(int argc, char** argv) {
    // ./hw3 [x1] [y1] [z1] [x2] [y2] [z2] [width] [height] [filename]
    assert(argc == 10);

    //--- init arguments
    // 使用 Host 端的結構體來收集參數
    RenderParams h_params; 
    
    h_params.camera_pos = make_double3(atof(argv[1]), atof(argv[2]), atof(argv[3]));
    h_params.target_pos = make_double3(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    
    unsigned int width = atoi(argv[7]);
    unsigned int height = atoi(argv[8]);

    h_params.iResolution = make_double2((double)width, (double)height);

    // 填入其他全域參數
    h_params.AA = AA;
    h_params.power = power;
    h_params.md_iter = (int)md_iter;
    h_params.ray_step = (int)ray_step;
    h_params.shadow_step = (int)shadow_step;
    h_params.step_limiter = step_limiter;
    h_params.ray_multiplier = ray_multiplier;
    h_params.bailout = bailout;
    h_params.eps = eps;
    h_params.FOV = FOV;
    h_params.far_plane = far_plane;
    //---

    printf("Initializing CUDA Mandelbulb Renderer...\n");
    printf("Resolution: %u x %u\n", width, height);

    //--- create host image buffer
    unsigned char* raw_image = new unsigned char[width * height * 4];
    
    // (原始碼中的 2D image** 陣列在 CUDA 版本中不再需要，
    // 因為 CPU 不會逐像素寫入)

    //--- CUDA setup ---
    unsigned char* d_image; // Device (GPU) image buffer
    size_t image_size = width * height * 4 * sizeof(unsigned char);

    // 1. 分配 GPU 記憶體
    printf("Allocating %lu bytes on GPU...\n", image_size);
    CUDA_CHECK(cudaMalloc(&d_image, image_size));

    // 2. 將渲染參數從 Host (h_params) 複製到 Device 的 __constant__ 記憶體 (d_params)
    CUDA_CHECK(cudaMemcpyToSymbol(d_params, &h_params, sizeof(RenderParams)));

    //--- Kernel launch setup ---
    
    // 3. 設定 Block 尺寸 (16x16 = 256 threads)
    // 這是基於 V100 (Warp=32) 的良好選擇
    dim3 blockDim(16, 16); 
    
    // 4. 設定 Grid 尺寸 (計算總共需要多少個 Block)
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                   (height + blockDim.y - 1) / blockDim.y);

    printf("Grid Dimensions: (%d, %d), Block Dimensions: (%d, %d)\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // 5. (可選) 使用 CUDA Events 測量 Kernel 執行時間
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    printf("Launching Kernel...\n");
    CUDA_CHECK(cudaEventRecord(start));

    //--- 6. 啟動 Kernel ---
    // (CPU 在此處將控制權交給 GPU)
    render_kernel<<<gridDim, blockDim>>>(d_image, width, height);

    //--- Post-kernel operations ---
    
    // 檢查 Kernel 啟動期間是否發生錯誤
    CUDA_CHECK(cudaGetLastError());
    
    // 7. 同步 CPU 和 GPU，並停止計時
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); // 等待 Kernel 完成

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    printf("Rendering complete. Copying image from GPU to CPU...\n");

    // 8. 將 GPU (d_image) 上的結果複製回 CPU (raw_image)
    CUDA_CHECK(cudaMemcpy(raw_image, d_image, image_size, cudaMemcpyDeviceToHost));

    //--- saving image
    printf("Saving image to: %s\n", argv[9]);
    write_png(argv[9], raw_image, width, height);
    //---

    //--- 9. finalize (釋放資源)
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_image)); // 釋放 GPU 記憶體
    delete[] raw_image;           // 釋放 CPU 記憶體
    //---

    printf("Done.\n");
    return 0;
}