#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
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

uint16_t floatToHalfBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = bits & 0x007FFFFFu;

    if (std::isnan(value)) {
        return static_cast<uint16_t>(sign | 0x7E00u);
    }

    if (std::isinf(value)) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }

        mantissa |= 0x00800000u;
        const uint32_t shiftedMantissa =
            mantissa >> static_cast<uint32_t>(1 - exponent);
        const uint32_t rounded = (shiftedMantissa + 0x00001000u) >> 13;
        return static_cast<uint16_t>(sign | rounded);
    }

    if (exponent >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    const uint32_t roundedMantissa = mantissa + 0x00001000u;
    if (roundedMantissa & 0x00800000u) {
        mantissa = 0;
        exponent += 1;
        if (exponent >= 0x1F) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
    } else {
        mantissa = roundedMantissa;
    }

    return static_cast<uint16_t>(
        sign |
        (static_cast<uint32_t>(exponent) << 10) |
        ((mantissa >> 13) & 0x03FFu));
}

uint16_t runSingleElementF16Dot(const std::string& ptxPath, uint16_t aBits, uint16_t bBits)
{
    PTXVM vm;
    EXPECT_TRUE(vm.initialize());
    EXPECT_TRUE(vm.loadProgram(ptxPath));

    const CUdeviceptr inputAPtr = vm.allocateMemory(sizeof(uint16_t));
    const CUdeviceptr inputBPtr = vm.allocateMemory(sizeof(uint16_t));
    const CUdeviceptr outputPtr = vm.allocateMemory(sizeof(uint16_t));
    EXPECT_NE(inputAPtr, 0u);
    EXPECT_NE(inputBPtr, 0u);
    EXPECT_NE(outputPtr, 0u);

    uint16_t outputBits = 0;
    EXPECT_TRUE(vm.copyMemoryHtoD(inputAPtr, &aBits, sizeof(uint16_t)));
    EXPECT_TRUE(vm.copyMemoryHtoD(inputBPtr, &bBits, sizeof(uint16_t)));
    EXPECT_TRUE(vm.copyMemoryHtoD(outputPtr, &outputBits, sizeof(uint16_t)));

    std::vector<KernelParameter> params;
    params.push_back({inputAPtr, sizeof(uint64_t), 0});
    params.push_back({inputBPtr, sizeof(uint64_t), 8});
    params.push_back({outputPtr, sizeof(uint64_t), 16});
    params.push_back({static_cast<CUdeviceptr>(1), sizeof(uint32_t), 24});

    vm.setKernelParameters(params);
    vm.getExecutor().setGridDimensions(1, 1, 1, 32, 1, 1);

    EXPECT_TRUE(vm.run());
    EXPECT_TRUE(vm.copyMemoryDtoH(&outputBits, outputPtr, sizeof(uint16_t)));

    vm.freeMemory(inputAPtr);
    vm.freeMemory(inputBPtr);
    vm.freeMemory(outputPtr);
    return outputBits;
}

}  // namespace

TEST(F16PTXCompatibilityTest, RunsBarraCUDAStyleHalfLoadConvertAndStore)
{
    const std::string ptx = R"(
.version 8.0
.target sm_89
.address_size 64

.entry solve (
    .param .u64 param0,
    .param .u64 param1,
    .param .u64 param2,
    .param .u32 param3
)
{
    .reg .u32  %r<7>;
    .reg .u64  %rd<9>;
    .reg .f32  %f<6>;
    .reg .pred %p<5>;
    .reg .f16  %h<4>;

$BB0:
    ld.param.u64 %rd1, [param0];
    ld.param.u64 %rd2, [param1];
    ld.param.u64 %rd3, [param2];
    ld.param.u32 %r1, [param3];
    mov.u32 %r2, %ctaid.x;
    setp.eq.u32 %p1, %r2, 0;
    @%p1 bra $BB1;
    setp.ne.u32 %p3, 0, 0;
    bra $BB2;
$BB1:
    mov.u32 %r3, %tid.x;
    setp.eq.u32 %p2, %r3, 0;
    selp.u32 %r6, 1, 0, %p2;
    setp.ne.u32 %p3, %r6, 0;
    bra $BB2;
$BB2:
    @%p3 bra $BB3;
    bra $BB4;
$BB3:
    mov.u32 %r4, 0;
    mov.f32 %f1, 0f00000000;
    bra $BB5;
$BB4:
    exit;
$BB5:
    setp.lt.s32 %p4, %r4, %r1;
    @%p4 bra $BB6;
    bra $BB8;
$BB6:
    cvt.u64.u32 %rd5, %r4;
    mad.lo.u64 %rd4, %rd5, 2, %rd1;
    ld.global.u32 %h1, [%rd4];
    cvt.f64.f32 %f3, %h1;
    cvt.u64.u32 %rd7, %r4;
    mad.lo.u64 %rd6, %rd7, 2, %rd2;
    ld.global.u32 %h2, [%rd6];
    cvt.f64.f32 %f4, %h2;
    mul.rn.f32 %f5, %f3, %f4;
    add.rn.f32 %f2, %f1, %f5;
    bra $BB7;
$BB7:
    add.u32 %r5, %r4, 1;
    mov.u32 %r4, %r5;
    mov.f32 %f1, %f2;
    bra $BB5;
$BB8:
    add.u64 %rd8, %rd3, 0;
    cvt.rn.f32.f64 %h3, %f1;
    st.global.u32 [%rd8], %h3;
    bra $BB4;
}
)";

    const std::string ptxPath = writeTempPTXFile(ptx);
    ASSERT_FALSE(ptxPath.empty());

    const uint16_t resultBits = runSingleElementF16Dot(
        ptxPath,
        floatToHalfBits(2.0f),
        floatToHalfBits(3.0f));

    EXPECT_EQ(resultBits, floatToHalfBits(6.0f));
    std::remove(ptxPath.c_str());
}

TEST(F16PTXCompatibilityTest, SupportsBitTypedHalfLoadsAndStores)
{
    const std::string ptx = R"(
.version 8.0
.target sm_89
.address_size 64

.entry solve (
    .param .u64 param0,
    .param .u64 param1,
    .param .u64 param2,
    .param .u32 param3
)
{
    .reg .pred %p<2>;
    .reg .u32 %r<2>;
    .reg .u64 %rd<6>;
    .reg .f16 %h<4>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [param0];
    ld.param.u64 %rd2, [param1];
    ld.param.u64 %rd3, [param2];
    ld.param.u32 %r1, [param3];

    mov.u32 %r1, %tid.x;
    setp.ne.u32 %p1, %r1, 0;
    @%p1 bra DONE;

    ld.global.b16 %h1, [%rd1];
    ld.global.b16 %h2, [%rd2];
    cvt.f32.f16 %f1, %h1;
    cvt.f32.f16 %f2, %h2;
    mul.rn.f32 %f3, %f1, %f2;
    cvt.rn.f16.f32 %h3, %f3;
    st.global.b16 [%rd3], %h3;

DONE:
    ret;
}
)";

    const std::string ptxPath = writeTempPTXFile(ptx);
    ASSERT_FALSE(ptxPath.empty());

    const uint16_t resultBits = runSingleElementF16Dot(
        ptxPath,
        floatToHalfBits(2.0f),
        floatToHalfBits(3.0f));

    EXPECT_EQ(resultBits, floatToHalfBits(6.0f));
    std::remove(ptxPath.c_str());
}
