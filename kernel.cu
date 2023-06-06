#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>
#include <algorithm>
#include <iostream>

#define nx 1024
#define ny 1024
#define BLOCK_SIZE 32
#define dt 0.1f
#define ds 0.1f
#define diffusion_constant 0.1f

#define check_error() if (cudaGetLastError() != cudaSuccess){std::cout << cudaGetErrorName(cudaGetLastError()) << " " << __FUNCTION__ << " " << __LINE__ << std::endl;exit(1);}

static float* u, * v, * p, * old_u, * old_v, * divergence, * gradX, * gradY, * r, * g, * b, * old_r, * old_g, * old_b;
static uint8_t* image;

__global__ void initializeVariables(float* u, float* v, float* p, float* old_u, float* old_v, float* divergence, float* gradX, float* gradY, float* r, float* g, float* b, float* old_r, float* old_g, float* old_b, uint8_t* image) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny)
    {
        int idx = j * nx + i;
        u[idx] = 10.0f;
        v[idx] = 10.0f;
        p[idx] = 0.0f;
        old_u[idx] = 10.0f;
        old_v[idx] = 10.0f;
        gradX[idx] = 0.0f;
        gradY[idx] = 0.0f;
        divergence[idx] = 0.0f;
        r[idx] = 0.0f;
        g[idx] = 0.0f;
        b[idx] = 0.0f;
        if (idx < (nx * ny / 2)) {
            old_r[idx] = 128.0f;
            old_g[idx] = 128.0f;
            old_b[idx] = 128.0f;
        }
        else {
            old_r[idx] = 0.0f;
            old_g[idx] = 0.0f;
            old_b[idx] = 0.0f;
        }
        image[idx * 4] = 0;
        image[idx * 4 + 1] = 0;
        image[idx * 4 + 2] = 0;
        image[idx * 4 + 3] = 0;
    }
}

void CUDA_INIT() {
    cudaMalloc(&u, nx * ny * sizeof(float));
    cudaMalloc(&v, nx * ny * sizeof(float));
    cudaMalloc(&p, nx * ny * sizeof(float));
    cudaMalloc(&old_u, nx * ny * sizeof(float));
    cudaMalloc(&old_v, nx * ny * sizeof(float));
    cudaMalloc(&divergence, nx * ny * sizeof(float));
    cudaMalloc(&gradX, nx * ny * sizeof(float));
    cudaMalloc(&gradY, nx * ny * sizeof(float));
    cudaMalloc(&r, nx * ny * sizeof(float));
    cudaMalloc(&g, nx * ny * sizeof(float));
    cudaMalloc(&b, nx * ny * sizeof(float));
    cudaMalloc(&old_r, nx * ny * sizeof(float));
    cudaMalloc(&old_g, nx * ny * sizeof(float));
    cudaMalloc(&old_b, nx * ny * sizeof(float));
    cudaMalloc(&image, 4 * nx * ny * sizeof(uint8_t));
    dim3 dimGrid(nx / BLOCK_SIZE, ny / BLOCK_SIZE, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    initializeVariables << <dimGrid, dimBlock >> > (u, v, p, old_u, old_v,divergence , gradX, gradY, r, g, b, old_r, old_g, old_b,image);
    cudaDeviceSynchronize();
    check_error();

}

void CUDA_EXIT() {
    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(old_u);
    cudaFree(old_v);
    cudaFree(divergence);
    cudaFree(gradX);
    cudaFree(gradY);
    cudaFree(r);
    cudaFree(g);
    cudaFree(b);
    cudaFree(old_r);
    cudaFree(old_g);
    cudaFree(old_b);
    cudaFree(image);
}

__device__ int clamp(int val, int min, int max) {
    return (val > max) ? max : ((val < min) ? min : val);
}
__device__ float clamp(float val, float min, float max) {
    return (val > max) ? max : ((val < min) ? min : val);
}

__global__ void kernel_advect(float* new_field, float* old_field, float* u, float* v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= nx || j >= ny) {
        return;
    }
    float x = float(i) * ds - dt * u[j * nx + i];
    float y = float(j) * ds - dt * v[j * nx + i];

    int i0 = clamp((int)(x / ds),0,nx-1);
    int j0 = clamp((int)(y / ds),0,ny-1);
    int i1 = i0 + 1;
    int j1 = j0 + 1;

    float sx = clamp(x / ds - i0,0.0f,1.0f);
    float sy = clamp(y / ds - j0,0.0f,1.0f);
    if (threadIdx.x == 0 && threadIdx.y==0) {
        printf("sx:%f\nsy:%f\n", sx, sy);
        printf("x:%f\ny:%f\n", x, y);
        printf("i0:%d\nj0:%d\ni0:%d\nj0:%d\n", i0, j0, i1, j1);
    }
    new_field[j * nx + i] = (1 - sx) * (1 - sy) * old_field[j0 * nx + i0] + (sx) * (1 - sy) * old_field[j0 * nx + i1] + (1 - sx) * (sy)*old_field[j1 * nx + i0] + (sx) * (sy) * old_field[j1 * nx + i1];
}

__global__ void kernel_diffusion(float* new_field,float* old_field) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;


    if (i >= nx || j >= ny) {
        return;
    }

    int idx = j * ny + i;

    float laplacian = 0.0f;

    int sampels = 0;

    if (i > 0) {
        sampels++;
        laplacian += old_field[idx - 1];
    }
    if (i < (nx - 1)) {
        sampels++;
        laplacian += old_field[idx + 1];
    }
    if (j > 0) {
        sampels++;
        laplacian += old_field[idx - nx];
    }
    if (j < (ny - 1)) {
        sampels++;
        laplacian += old_field[idx + nx];
    }

    laplacian -= sampels * old_field[idx];

    new_field[idx] = old_field[idx] + diffusion_constant * dt * laplacian;
}

__global__ void compute_divergence(float* u, float* v, float* divergence) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < nx) && (j < ny)) {
        int index = i + j * ny;
        float u_right = (i == nx - 1) ? 0.0f : u[index + 1];
        float u_left = (i == 0) ? 0.0f : u[index - 1];
        float v_up = (j == ny - 1) ? 0.0f : v[index + nx];
        float v_down = (j == 0) ? 0.0f : v[index - nx];
        float dx_u = (u_right - u_left) * 1/ds;
        float dy_v = (v_up - v_down) * 1/ds;
        divergence[index] = dx_u + dy_v;
    }
}

__global__ void jacobi(float* preassure, float* divergence) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i > 0) && (i < (nx - 1)) && (j > 0) && (j < (ny - 1))) {
        int index = i + j * nx;
        preassure[index] = (divergence[index] + preassure[index - 1] + preassure[index + 1] + preassure[index - ny] + preassure[index + nx]) * 0.25f;
    }
}

__global__ void kernel_gradient(float* field, float* gradX, float* gradY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i > 0) && (i < (nx - 1)) && (j > 0) && (j < (ny - 1))) {
        gradX[j * nx + i] = (field[j * nx + i + 1] - field[j * nx + i - 1]) / (2.0f * ds);
        gradY[j * nx + i] = (field[(j + 1) * nx + i] - field[(j - 1) * nx + i]) / (2.0f * ds);
    }
}

__global__ void subtract_arrays(float* a, float* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((j * nx + i) < nx*ny) {
        a[j * nx + i] = a[j * nx + i] - b[j * nx + i];
    }
}

void preasure_projection(dim3 dimGrid, dim3 dimBlock) {
    compute_divergence<<<dimGrid,dimBlock>>>(u, v, divergence);
    for (int i = 0; i < 40; i++) {
        jacobi << <dimGrid, dimBlock >> > (p, divergence);
    }
    kernel_gradient << <dimGrid, dimBlock >> > (p,gradX,gradY);
    subtract_arrays << <dimGrid, dimBlock >> > (u, gradX);
    subtract_arrays << <dimGrid, dimBlock >> > (v, gradY);
    cudaDeviceSynchronize();
    check_error();

}

void advect(dim3 dimGrid, dim3 dimBlock) {
    kernel_advect << <dimGrid, dimBlock >> > (u, old_u, old_u, old_v);
    cudaDeviceSynchronize();
    check_error();
    kernel_advect << <dimGrid, dimBlock >> > (v, old_v, old_u ,old_v);
    cudaDeviceSynchronize();
    check_error();
    kernel_advect << <dimGrid, dimBlock >> > (r, old_r, u, v);
    cudaDeviceSynchronize();
    check_error();
    kernel_advect << <dimGrid, dimBlock >> > (g, old_g, u, v);
    cudaDeviceSynchronize();
    check_error();
    kernel_advect << <dimGrid, dimBlock >> > (b, old_b, u, v);
    cudaDeviceSynchronize();
    check_error();
}

void diffusion(dim3 dimGrid, dim3 dimBlock) {
    kernel_diffusion << <dimGrid, dimBlock >> > (u, old_u);
    kernel_diffusion << <dimGrid, dimBlock >> > (v, old_v);
    kernel_diffusion << <dimGrid, dimBlock >> > (r, old_r);
    kernel_diffusion << <dimGrid, dimBlock >> > (g, old_g);
    kernel_diffusion << <dimGrid, dimBlock >> > (b, old_b);
    cudaDeviceSynchronize();
    check_error();
}

__global__ void kernel_create_color(float* r, float* g, float* b, uint8_t* image) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    image[(j * nx + i) * 4] = (r[j * nx + i] < 255) ? (uint8_t)r[j * nx + i] : (uint8_t)255;
    image[(j * nx + i) * 4+1] = (g[j * nx + i] < 255) ? (uint8_t)g[j * nx + i] : (uint8_t)255;
    image[(j * nx + i) * 4+2] = (b[j * nx + i] < 255) ? (uint8_t)b[j * nx + i] : (uint8_t)255;
    image[(j * nx + i) * 4+3] = (uint8_t)255;
}

void create_color(dim3 dimGrid, dim3 dimBlock) {
    kernel_create_color<<<dimGrid,dimBlock>>>(r, g, b, image);
    cudaDeviceSynchronize();
    check_error();
}

void COMPUTE_FIELD(uint8_t* result) {
    dim3 dimGrid(nx / BLOCK_SIZE, ny / BLOCK_SIZE, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    advect(dimGrid, dimBlock);
    diffusion(dimGrid, dimBlock);
    preasure_projection(dimGrid, dimBlock);
    std::swap(u, old_u);
    std::swap(v, old_v);
    std::swap(r, old_r);
    std::swap(g, old_g);
    std::swap(b, old_b);
    create_color(dimGrid, dimBlock);
    cudaMemcpy(result, image, 4 * ny * nx * sizeof(uint8_t), cudaMemcpyDeviceToHost);
}
