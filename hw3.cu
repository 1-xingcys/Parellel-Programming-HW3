#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// CUDA 執行時標頭檔
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 用於 device 端的向量類型 (如 float3, float2)
#include <vector_types.h>

// LodePNG 用於儲存影像
#include <lodepng.h>

// 移除 GLM 包含

#define pi 3.14159265358979323846

//==============================================================================
// CUDA Device-Side Helper Functions
// (用以取代 GLM 的功能)
//==============================================================================

// --- Clamp (fmin/fmax) ---
// 使用 __device__ __forceinline__ 建議編譯器將其內聯
__device__ __forceinline__ float clamp(float val, float min, float max) {
    return fmin(fmax(val, min), max);
}

__device__ __forceinline__ float3 clamp(float3 val, float min, float max) {
    return make_float3(clamp(val.x, min, max),
                        clamp(val.y, min, max),
                        clamp(val.z, min, max));
}


// --- float3 的運算子重載 ---
__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 operator+(float3 a, float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}
__device__ __forceinline__ void operator+=(float3 &a, float3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}
__device__ __forceinline__ void operator+=(float3 &a, float b) {
    a.x += b; a.y += b; a.z += b;
}
__device__ __forceinline__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float3 operator-(float3 a) { // Unary minus
    return make_float3(-a.x, -a.y, -a.z);
}
__device__ __forceinline__ float3 operator*(float3 a, float s) { // vec * scalar
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float3 operator*(float s, float3 a) { // scalar * vec
    return make_float3(a.x * s, a.y * s, a.z * s); // << 修正： b10502010 感謝您，這裡修正 a.Y -> a.y
}
__device__ __forceinline__ float3 operator*(float3 a, float3 b) { // component-wise
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__device__ __forceinline__ void operator*=(float3 &a, float3 b) { // component-wise
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}
__device__ __forceinline__ float3 operator/(float3 a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

// --- float2 的運算子重載 ---
__device__ __forceinline__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ float2 operator-(float2 a) { // Unary minus
    return make_float2(-a.x, -a.y);
}
__device__ __forceinline__ float2 operator*(float s, float2 a) { // scalar * vec
    return make_float2(a.x * s, a.y * s);
}
__device__ __forceinline__ float2 operator/(float2 a, float s) {
    return make_float2(a.x / s, a.y / s);
}

// --- 向量數學函式 ---
__device__ __forceinline__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float length(float3 v) {
    return sqrt(dot(v, v));
}
__device__ __forceinline__ float3 normalize(float3 v) {
    // 避免除以零 (如果 length 為 0，回傳 (0,0,0) 避免 NaN)
    float l = length(v);
    if (l == 0.0) return make_float3(0.0, 0.0, 0.0);
    float invLen = 1.0 / l;
    return v * invLen;
}
__device__ __forceinline__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y,
                        a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x);
}
// 針對 pow(vec, vec) 的逐元素 (component-wise) pow
__device__ __forceinline__ float3 pow(float3 a, float3 b) {
    return make_float3(pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z));
}
// 針對 pow(vec, scalar) 的逐元素 pow
__device__ __forceinline__ float3 pow(float3 a, float b) {
    return make_float3(pow(a.x, b), pow(a.y, b), pow(a.z, b));
}

// -----------------------------------------------------------------------------

// 原始 C++ 程式碼中的全域變數，現在移入此結構體
struct RenderParams {
    float power;
    int md_iter;
    int ray_step;
    int shadow_step;
    float step_limiter;
    float ray_multiplier;
    float bailout;
    float eps;
    float FOV;
    float far_plane;
    float3 camera_pos;
    float3 target_pos;
    float2 iResolution;
    int AA;
};

// 在 GPU 的常數記憶體中宣告一個 RenderParams 實例
// __constant__ 記憶體會被快取並高效廣播
__constant__ RenderParams d_params;

// 原始 CPU 程式碼中的變數 (用於 main 函式初始化)
int AA = 3;
float power = 8.0;
float md_iter = 24;
float ray_step = 10000;
float shadow_step = 1500;
float step_limiter = 0.2;
float ray_multiplier = 0.1;
float bailout = 2.0;
float eps = 0.0005;
float FOV = 1.5;
float far_plane = 100.;


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
__device__ float md(float3 p, float& trap) {
    float3 v = p;
    float dr = 1.;
    float r = length(v); 
    trap = r;

    // 使用常數記憶體中的參數
    for (int i = 0; i < d_params.md_iter; ++i) {
        float theta = atan2(v.y, v.x) * d_params.power;
        float phi = asin(v.z / r) * d_params.power;
        float s_th, c_th, s_ph, c_ph;
        sincos(theta, &s_th, &c_th);
        sincos(phi,   &s_ph, &c_ph);

        float r2 = r * r;
        float r4 = r2 * r2;
        float r8 = r4 * r4;
        float r7 = r8 / r;
        
        dr = d_params.power * r7 * dr + 1.;
        
        float3 v_new = make_float3(c_th * c_ph, 
                                    c_ph * s_th, 
                                    -s_ph);
        v = p + r8 * v_new;

        trap = fmin(trap, r);
        r = length(v);
        if (r > d_params.bailout) break;
    }
    return 0.5 * log(r) * r / dr;
}

// scene mapping
__device__ float map(float3 p, float& trap, int& ID) {
    // 原始碼的矩陣乘法是繞 X 軸旋轉 90 度
    // (x, y, z) -> (x, z, -y)
    float3 rp = make_float3(p.x, p.z, -p.y);
    ID = 1;
    return md(rp, trap);
}

// dummy function overload
__device__ float map(float3 p) {
    float dmy;
    int dmy2;
    return map(p, dmy, dmy2);
}

// simple palette function (borrowed from Inigo Quilez)
__device__ float3 pal(float t, float3 a, float3 b, float3 c, float3 d) {
    float cos_val_r = cos(2. * pi * (c.x * t + d.x));
    float cos_val_g = cos(2. * pi * (c.y * t + d.y));
    float cos_val_b = cos(2. * pi * (c.z * t + d.z));
    
    return a + b * make_float3(cos_val_r, cos_val_g, cos_val_b);
}

// second march: cast shadow
__device__ float softshadow(float3 ro, float3 rd, float k) {
    float res = 1.0;
    float t = 0.;
    float small_eps = 1e-6;
    for (int i = 0; i < d_params.shadow_step; ++i) {
        float h = map(ro + rd * t);
        float denom = fmax(t, small_eps);
        res = fmin(res, k * h / denom); 
        if (res < 0.02) return 0.02;
        if (t > d_params.far_plane) break;
        // 使用我們自己的 clamp 輔助函式
        t += clamp(h, .001, d_params.step_limiter);
    }
    return clamp(res, .02, 1.);
}

// use gradient to calc surface normal
__device__ float3 calcNor(float3 p) {
    float2 e = make_float2(d_params.eps, 0.0);
    
    float3 e_xyy = make_float3(e.x, e.y, e.y); // (eps, 0, 0)
    float3 e_yxy = make_float3(e.y, e.x, e.y); // (0, eps, 0)
    float3 e_yyx = make_float3(e.y, e.y, e.x); // (0, 0, eps)

    return normalize(make_float3( 
        map(p + e_xyy) - map(p - e_xyy),
        map(p + e_yxy) - map(p - e_yxy),
        map(p + e_yyx) - map(p - e_yyx)
    ));
}

// first march: find object's surface
__device__ float trace(float3 ro, float3 rd, float& trap, int& ID) {
    float t = 0;
    float len = 0;

    for (int i = 0; i < d_params.ray_step; ++i) {
        len = map(ro + rd * t, trap, ID);
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
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 像素 x 座標 (width)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 像素 y 座標 (height)

    // 邊界檢查：防止執行緒處理超出影像範圍的像素
    if (j >= width || i >= height) {
        return;
    }

    // --- 程式碼主體：從 CPU main() 迴圈中複製而來 ---
    
    float fcol_r = 0.0;
    float fcol_g = 0.0;
    float fcol_b = 0.0;

    // create camera
    float3 ro = d_params.camera_pos;
    float3 ta = d_params.target_pos;
    float3 cf = normalize(ta - ro); 
    float3 cs = normalize(cross(cf, make_float3(0., 1., 0.))); 
    float3 cu = normalize(cross(cs, cf));

    // anti aliasing
    for (int m = 0; m < d_params.AA; ++m) {
        for (int n = 0; n < d_params.AA; ++n) {
            
            float2 p = make_float2((float)j, (float)i) + 
                       make_float2((float)m, (float)n) / (float)d_params.AA;

            // vec2 uv = ... (使用 d_params.iResolution)
            // (原始碼中 -iResolution.xy() + 2. * p)
            float2 uv = (make_float2(2.*p.x, 2.*p.y) - d_params.iResolution) / d_params.iResolution.y;
            uv.y *= -1; // Y 軸翻轉

            float3 rd = normalize(uv.x * cs + uv.y * cu + d_params.FOV * cf);

            // marching
            float trap;
            int objID;
            float d = trace(ro, rd, trap, objID);

            // lighting
            float3 col = make_float3(0.0, 0.0, 0.0);
            float3 sd = normalize(d_params.camera_pos); // Sun direction
            float3 sc = make_float3(1., .9, .717);     // Light color

            // coloring
            if (d < 0.) { // miss (hit sky)
                col = make_float3(0.0, 0.0, 0.0); // sky color (black)
            } else {
                float3 pos = ro + rd * d;
                float3 nr = calcNor(pos);
                float3 hal = normalize(sd - rd); // blinn-phong

                col = pal(trap - .4, make_float3(.5, .5, .5), make_float3(.5, .5, .5), 
                          make_float3(1., 1., 1.), make_float3(.0, .1, .2));
                float3 ambc = make_float3(0.3, 0.3, 0.3);
                float gloss = 32.;

                // simple blinn phong
                float amb = (0.7 + 0.3 * nr.y) * (0.2 + 0.8 * clamp(0.05 * log(trap), 0.0, 1.0));
                float sdw = softshadow(pos + 0.001 * nr, sd, 16.);
                float dif = clamp(dot(sd, nr), 0., 1.) * sdw;
                float spe = pow(clamp(dot(nr, hal), 0., 1.), gloss) * dif;

                float3 lin = make_float3(0.0, 0.0, 0.0);
                lin += ambc * (.05 + .95 * amb);
                lin += sc * dif * 0.8;
                col *= lin;

                // 逐元素 pow
                col = pow(col, make_float3(0.7, 0.9, 1.0));
                col += spe * 0.8;
            }

            // Gamma correction 
            // 這裡現在會呼叫我們新增的 clamp(float3, float, float) 重載
            col = clamp(pow(col, make_float3(0.4545, 0.4545, 0.4545)), 0.0, 1.0);
            
            fcol_r += col.x; 
            fcol_g += col.y; 
            fcol_b += col.z; 
        }
    }
    // --- 迴圈結束 ---

    fcol_r /= (float)(d_params.AA * d_params.AA);
    fcol_g /= (float)(d_params.AA * d_params.AA);
    fcol_b /= (float)(d_params.AA * d_params.AA);
    
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
    RenderParams h_params; 
    
    h_params.camera_pos = make_float3((float)atof(argv[1]), (float)atof(argv[2]), (float)atof(argv[3]));
    h_params.target_pos = make_float3((float)atof(argv[4]), (float)atof(argv[5]), (float)atof(argv[6]));
    
    unsigned int width = atoi(argv[7]);
    unsigned int height = atoi(argv[8]);

    h_params.iResolution = make_float2((float)width, (float)height);

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

    // printf("Initializing CUDA Mandelbulb Renderer...\n");
    // printf("Resolution: %u x %u\n", width, height);

    //--- create host image buffer
    unsigned char* raw_image = new unsigned char[width * height * 4];
    
    //--- CUDA setup ---
    unsigned char* d_image; // Device (GPU) image buffer
    size_t image_size = width * height * 4 * sizeof(unsigned char);

    // 1. 分配 GPU 記憶體
    // printf("Allocating %lu bytes on GPU...\n", image_size);
    CUDA_CHECK(cudaMalloc(&d_image, image_size));

    // 2. 將渲染參數從 Host (h_params) 複製到 Device 的 __constant__ 記憶體 (d_params)
    CUDA_CHECK(cudaMemcpyToSymbol(d_params, &h_params, sizeof(RenderParams)));

    //--- Kernel launch setup ---
    
    // 3. 設定 Block 尺寸 (16x16 = 256 threads)
    dim3 blockDim(16, 16); 
    
    // 4. 設定 Grid 尺寸
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                   (height + blockDim.y - 1) / blockDim.y);

    // printf("Grid Dimensions: (%d, %d), Block Dimensions: (%d, %d)\n", 
        //    gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // 5. 使用 CUDA Events 測量 Kernel 執行時間
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // printf("Launching Kernel...\n");
    CUDA_CHECK(cudaEventRecord(start));

    //--- 6. 啟動 Kernel ---
    render_kernel<<<gridDim, blockDim>>>(d_image, width, height);

    //--- Post-kernel operations ---
    CUDA_CHECK(cudaGetLastError());
    
    // 7. 同步 CPU 和 GPU，並停止計時
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); 

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    // printf("Rendering complete. Copying image from GPU to CPU...\n");

    // 8. 將 GPU (d_image) 上的結果複製回 CPU (raw_image)
    CUDA_CHECK(cudaMemcpy(raw_image, d_image, image_size, cudaMemcpyDeviceToHost));

    //--- saving image
    // printf("Saving image to: %s\n", argv[9]);
    write_png(argv[9], raw_image, width, height);
    //---

    //--- 9. finalize (釋放資源)
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_image)); 
    delete[] raw_image;           
    //---

    // printf("Done.\n");
    return 0;
}