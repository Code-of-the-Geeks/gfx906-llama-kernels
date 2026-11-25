#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
__device__ float dec(uint8_t x){ const uint32_t u=x&0x7F; const float norm=__uint_as_float((u<<20)+(119u<<23)); const float sub=(float)(u&7)*0.0009765625f; const float r=(u>>3)?norm:sub; return x==0x7F?0.0f:r; }
__global__ void k(float*o){ o[threadIdx.x]=dec(threadIdx.x); }
static float ref(uint8_t x){ if(x==0||x==0x7F) return 0; int e=(x>>3)&0xF,m=x&7; float r= e==0? ldexpf(m,-9): ldexpf(1+m/8.0f,e-7); return r*0.5f; }
int main(){ float*d; hipMalloc(&d,1024); k<<<1,256>>>(d); float h[256]; hipMemcpy(h,d,1024,hipMemcpyDeviceToHost); int bad=0; for(int i=0;i<256;i++){ float r=ref(i); if(memcmp(&r,&h[i],4)){bad++; printf("%d %g %g\n",i,r,h[i]);} } printf("mismatches %d/256\n",bad); }
