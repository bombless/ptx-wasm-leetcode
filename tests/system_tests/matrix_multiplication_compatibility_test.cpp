#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "vm.hpp"

namespace {

std::string writeTempPTXFile(const std::string& contents)
{
    std::array<char, L_tmpnam> pathBuffer {};
    char* rawPath = std::tmpnam(pathBuffer.data());
    EXPECT_NE(rawPath, nullptr);

    std::string path = rawPath ? rawPath : "";
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

float runSingleMatrixOutput(const std::string& ptxPath,
                            const std::vector<float>& inputA,
                            const std::vector<float>& inputB,
                            int rowsM,
                            int sharedN,
                            int colsK,
                            int row,
                            int col)
{
    PTXVM vm;
    EXPECT_TRUE(vm.initialize());
    EXPECT_TRUE(vm.loadProgram(ptxPath));

    const size_t outputElements = static_cast<size_t>(rowsM) * static_cast<size_t>(colsK);
    std::vector<float> output(outputElements, 0.0f);

    const size_t inputABytes = inputA.size() * sizeof(float);
    const size_t inputBBytes = inputB.size() * sizeof(float);
    const size_t outputBytes = output.size() * sizeof(float);

    const CUdeviceptr inputAPtr = vm.allocateMemory(inputABytes);
    const CUdeviceptr inputBPtr = vm.allocateMemory(inputBBytes);
    const CUdeviceptr outputPtr = vm.allocateMemory(outputBytes);
    EXPECT_NE(inputAPtr, 0u);
    EXPECT_NE(inputBPtr, 0u);
    EXPECT_NE(outputPtr, 0u);

    EXPECT_TRUE(vm.copyMemoryHtoD(inputAPtr, inputA.data(), inputABytes));
    EXPECT_TRUE(vm.copyMemoryHtoD(inputBPtr, inputB.data(), inputBBytes));
    EXPECT_TRUE(vm.copyMemoryHtoD(outputPtr, output.data(), outputBytes));

    std::vector<KernelParameter> params;
    params.push_back({inputAPtr, sizeof(uint64_t), 0});
    params.push_back({inputBPtr, sizeof(uint64_t), 8});
    params.push_back({outputPtr, sizeof(uint64_t), 16});
    params.push_back({static_cast<CUdeviceptr>(rowsM), sizeof(uint32_t), 24});
    params.push_back({static_cast<CUdeviceptr>(sharedN), sizeof(uint32_t), 28});
    params.push_back({static_cast<CUdeviceptr>(colsK), sizeof(uint32_t), 32});

    vm.setKernelParameters(params);

    constexpr unsigned int kBlockDimX = 16;
    constexpr unsigned int kBlockDimY = 16;
    const unsigned int gridDimX =
        static_cast<unsigned int>((colsK + static_cast<int>(kBlockDimX) - 1) / static_cast<int>(kBlockDimX));
    const unsigned int gridDimY =
        static_cast<unsigned int>((rowsM + static_cast<int>(kBlockDimY) - 1) / static_cast<int>(kBlockDimY));

    PTXExecutor& executor = vm.getExecutor();
    executor.setGridDimensions(gridDimX, gridDimY, 1, kBlockDimX, kBlockDimY, 1);

    ThreadExecutionContext context;
    context.gridDimX = gridDimX;
    context.gridDimY = gridDimY;
    context.gridDimZ = 1;
    context.blockDimX = kBlockDimX;
    context.blockDimY = kBlockDimY;
    context.blockDimZ = 1;
    context.blockIdxX = static_cast<unsigned int>(col / static_cast<int>(kBlockDimX));
    context.blockIdxY = static_cast<unsigned int>(row / static_cast<int>(kBlockDimY));
    context.blockIdxZ = 0;
    context.threadIdxX = static_cast<unsigned int>(col % static_cast<int>(kBlockDimX));
    context.threadIdxY = static_cast<unsigned int>(row % static_cast<int>(kBlockDimY));
    context.threadIdxZ = 0;
    context.warpSize = 32;
    context.laneId = static_cast<unsigned int>(
        ((context.threadIdxY * kBlockDimX) + context.threadIdxX) % context.warpSize);
    executor.setSingleThreadExecutionContext(context);

    EXPECT_TRUE(vm.run());

    float outputValue = 0.0f;
    const size_t outputIndex =
        static_cast<size_t>(row) * static_cast<size_t>(colsK) + static_cast<size_t>(col);
    EXPECT_TRUE(vm.copyMemoryDtoH(
        &outputValue,
        outputPtr + static_cast<CUdeviceptr>(outputIndex * sizeof(float)),
        sizeof(float)));

    vm.freeMemory(inputAPtr);
    vm.freeMemory(inputBPtr);
    vm.freeMemory(outputPtr);
    return outputValue;
}

}  // namespace

TEST(MatrixMultiplicationCompatibilityTest, RunsBarraCUDAGeneratedKernel)
{
    const std::string ptx = R"(
.version 8.0
.target sm_89
.address_size 64

.entry solve (
    .param .u64 param0,
    .param .u64 param1,
    .param .u64 param2,
    .param .u32 param3,
    .param .u32 param4,
    .param .u32 param5
)
{
    .reg .u32  %r<23>;
    .reg .u64  %rd<10>;
    .reg .f32  %f<6>;
    .reg .pred %p<5>;

$BB0:
    ld.param.u64 %rd1, [param0];
    ld.param.u64 %rd2, [param1];
    ld.param.u64 %rd3, [param2];
    ld.param.u32 %r1, [param3];
    ld.param.u32 %r2, [param4];
    ld.param.u32 %r3, [param5];
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mul.lo.u32 %r6, %r4, %r5;
    mov.u32 %r7, %tid.x;
    add.u32 %r8, %r6, %r7;
    mov.u32 %r9, %ctaid.y;
    mov.u32 %r10, %ntid.y;
    mul.lo.u32 %r11, %r9, %r10;
    mov.u32 %r12, %tid.y;
    add.u32 %r13, %r11, %r12;
    setp.ge.s32 %p1, %r13, %r1;
    @%p1 bra $BB9;
    bra $BB1;
$BB1:
    setp.ge.s32 %p2, %r8, %r3;
    selp.u32 %r22, 1, 0, %p2;
    setp.ne.u32 %p3, %r22, 0;
    bra $BB2;
$BB2:
    @%p3 bra $BB3;
    bra $BB4;
$BB3:
    exit;
$BB4:
    mov.u32 %r14, 0;
    mov.f32 %f1, 0f00000000;
    bra $BB5;
$BB5:
    setp.lt.s32 %p4, %r14, %r2;
    @%p4 bra $BB6;
    bra $BB8;
$BB6:
    mul.lo.u32 %r16, %r13, %r2;
    add.u32 %r17, %r16, %r14;
    cvt.u64.u32 %rd5, %r17;
    mad.lo.u64 %rd4, %rd5, 4, %rd1;
    ld.global.f32 %f3, [%rd4];
    mul.lo.u32 %r18, %r14, %r3;
    add.u32 %r19, %r18, %r8;
    cvt.u64.u32 %rd7, %r19;
    mad.lo.u64 %rd6, %rd7, 4, %rd2;
    ld.global.f32 %f4, [%rd6];
    mul.rn.f32 %f5, %f3, %f4;
    add.rn.f32 %f2, %f1, %f5;
    bra $BB7;
$BB7:
    add.u32 %r15, %r14, 1;
    mov.u32 %r14, %r15;
    mov.f32 %f1, %f2;
    bra $BB5;
$BB8:
    mul.lo.u32 %r20, %r13, %r3;
    add.u32 %r21, %r20, %r8;
    cvt.u64.u32 %rd9, %r21;
    mad.lo.u64 %rd8, %rd9, 4, %rd3;
    st.global.f32 [%rd8], %f1;
    exit;
$BB9:
    setp.ne.u32 %p3, 1, 0;
    bra $BB2;
}
)";

    const std::string ptxPath = writeTempPTXFile(ptx);
    ASSERT_FALSE(ptxPath.empty());

    const std::vector<float> inputA = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    const std::vector<float> inputB = {
        5.0f, 6.0f,
        7.0f, 8.0f,
    };

    std::array<float, 4> output {};
    output[0] = runSingleMatrixOutput(ptxPath, inputA, inputB, 2, 2, 2, 0, 0);
    output[1] = runSingleMatrixOutput(ptxPath, inputA, inputB, 2, 2, 2, 0, 1);
    output[2] = runSingleMatrixOutput(ptxPath, inputA, inputB, 2, 2, 2, 1, 0);
    output[3] = runSingleMatrixOutput(ptxPath, inputA, inputB, 2, 2, 2, 1, 1);

    EXPECT_FLOAT_EQ(output[0], 19.0f);
    EXPECT_FLOAT_EQ(output[1], 22.0f);
    EXPECT_FLOAT_EQ(output[2], 43.0f);
    EXPECT_FLOAT_EQ(output[3], 50.0f);

    std::remove(ptxPath.c_str());
}
