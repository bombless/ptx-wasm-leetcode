#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "parser/parser.hpp"
#include "vm.hpp"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

namespace {

struct ParamSummary {
    std::string name;
    std::string type;
    size_t offset = 0;
    size_t size = 0;
    bool isPointer = false;
};

struct EntrySummary {
    std::string name;
    std::vector<ParamSummary> params;
};

class WebPTXBridge {
public:
    bool loadProgram(const std::string& path) {
        clearProgramState();

        PTXParser parser;
        if (!parser.parseFile(path)) {
            lastError_ = parser.getErrorMessage().empty()
                ? "Failed to parse PTX file."
                : parser.getErrorMessage();
            return false;
        }

        const PTXProgram& program = parser.getProgram();
        for (size_t entryIndex : program.entryPoints) {
            if (entryIndex >= program.functions.size()) {
                continue;
            }

            const PTXFunction& function = program.functions[entryIndex];
            EntrySummary entry;
            entry.name = function.name;

            for (const PTXParameter& param : function.parameters) {
                entry.params.push_back({
                    param.name,
                    param.type,
                    param.offset,
                    param.size,
                    param.isPointer,
                });
            }

            entries_.push_back(std::move(entry));
        }

        if (entries_.empty()) {
            lastError_ = "The PTX file loaded, but no entry kernels were found.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(path)) {
            lastError_ = "The PTX file parsed, but the VM could not load it.";
            return false;
        }

        loadedPath_ = path;
        return true;
    }

    int getEntryCount() const {
        return static_cast<int>(entries_.size());
    }

    const char* getEntryName(int entryIndex) {
        if (entryIndex < 0 || static_cast<size_t>(entryIndex) >= entries_.size()) {
            scratch_ = "";
            return scratch_.c_str();
        }

        scratch_ = entries_[static_cast<size_t>(entryIndex)].name;
        return scratch_.c_str();
    }

    int getParamCount(int entryIndex) const {
        if (entryIndex < 0 || static_cast<size_t>(entryIndex) >= entries_.size()) {
            return 0;
        }

        return static_cast<int>(entries_[static_cast<size_t>(entryIndex)].params.size());
    }

    const char* getParamName(int entryIndex, int paramIndex) {
        const ParamSummary* param = getParam(entryIndex, paramIndex);
        scratch_ = param ? param->name : "";
        return scratch_.c_str();
    }

    const char* getParamType(int entryIndex, int paramIndex) {
        const ParamSummary* param = getParam(entryIndex, paramIndex);
        scratch_ = param ? param->type : "";
        return scratch_.c_str();
    }

    int isParamPointer(int entryIndex, int paramIndex) const {
        const ParamSummary* param = getParam(entryIndex, paramIndex);
        return (param != nullptr && param->isPointer) ? 1 : 0;
    }

    bool runVectorAddDemo(const std::string& kernelName,
                          const float* inputA,
                          int inputALen,
                          const float* inputB,
                          int inputBLen) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputA == nullptr || inputB == nullptr) {
            lastError_ = "Input buffers must not be null.";
            return false;
        }

        if (inputALen <= 0 || inputBLen <= 0) {
            lastError_ = "Both input arrays must contain at least one float.";
            return false;
        }

        if (inputALen != inputBLen) {
            lastError_ = "Input A and Input B must have the same length.";
            return false;
        }

        const size_t elementCount = static_cast<size_t>(inputALen);
        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isVectorAddSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32), such as vector_add.";
            return false;
        }

        // The current PTX VM executes one representative lane per warp. To keep
        // standard thread-indexed vector-add kernels usable in the browser judge,
        // replay the kernel once per output element using one-float slices.
        lastResult_.assign(elementCount, 0.0f);

        for (size_t i = 0; i < elementCount; ++i) {
            auto vm = createVm();
            if (!vm) {
                return false;
            }

            if (!vm->loadProgram(loadedPath_)) {
                lastError_ = "Failed to reload the uploaded PTX program.";
                lastResult_.clear();
                return false;
            }

            const size_t bytes = sizeof(float);
            const CUdeviceptr inputAPtr = vm->allocateMemory(bytes);
            const CUdeviceptr inputBPtr = vm->allocateMemory(bytes);
            const CUdeviceptr outputPtr = vm->allocateMemory(bytes);

            if (!vm->copyMemoryHtoD(inputAPtr, &inputA[i], bytes)) {
                lastError_ = "Failed to copy Input A into VM memory.";
                lastResult_.clear();
                return false;
            }

            if (!vm->copyMemoryHtoD(inputBPtr, &inputB[i], bytes)) {
                lastError_ = "Failed to copy Input B into VM memory.";
                lastResult_.clear();
                return false;
            }

            std::vector<KernelParameter> params;
            params.push_back({inputAPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
            params.push_back({inputBPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
            params.push_back({outputPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
            params.push_back({
                static_cast<CUdeviceptr>(1),
                kernel->parameters[3].size,
                kernel->parameters[3].offset,
            });

            vm->setKernelParameters(params);
            vm->getExecutor().setGridDimensions(1, 1, 1, 32, 1, 1);

            if (!vm->run()) {
                lastError_ = "Kernel execution failed inside the PTX VM.";
                lastResult_.clear();
                return false;
            }

            float outputValue = 0.0f;
            if (!vm->copyMemoryDtoH(&outputValue, outputPtr, bytes)) {
                lastError_ = "Kernel executed, but reading the result buffer failed.";
                lastResult_.clear();
                return false;
            }

            lastResult_[i] = outputValue;
        }

        return true;
    }

    bool runMatrixMultiplicationDemo(const std::string& kernelName,
                                     const float* inputA,
                                     int inputALen,
                                     const float* inputB,
                                     int inputBLen,
                                     int rowsM,
                                     int sharedN,
                                     int colsK) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputA == nullptr || inputB == nullptr) {
            lastError_ = "Input buffers must not be null.";
            return false;
        }

        if (rowsM <= 0 || sharedN <= 0 || colsK <= 0) {
            lastError_ = "Matrix dimensions M, N, and K must all be positive integers.";
            return false;
        }

        const size_t expectedALen =
            static_cast<size_t>(rowsM) * static_cast<size_t>(sharedN);
        const size_t expectedBLen =
            static_cast<size_t>(sharedN) * static_cast<size_t>(colsK);
        const size_t outputElementCount =
            static_cast<size_t>(rowsM) * static_cast<size_t>(colsK);

        if (static_cast<size_t>(inputALen) != expectedALen) {
            lastError_ = "Matrix A does not match the provided M x N dimensions.";
            return false;
        }

        if (static_cast<size_t>(inputBLen) != expectedBLen) {
            lastError_ = "Matrix B does not match the provided N x K dimensions.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isMatrixMultiplySignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32, .u32, .u32), such as matrix multiplication.";
            return false;
        }

        constexpr unsigned int kBlockDimX = 16;
        constexpr unsigned int kBlockDimY = 16;
        const unsigned int gridDimX =
            static_cast<unsigned int>((colsK + static_cast<int>(kBlockDimX) - 1) / static_cast<int>(kBlockDimX));
        const unsigned int gridDimY =
            static_cast<unsigned int>((rowsM + static_cast<int>(kBlockDimY) - 1) / static_cast<int>(kBlockDimY));

        const size_t inputABytes = expectedALen * sizeof(float);
        const size_t inputBBytes = expectedBLen * sizeof(float);
        const size_t outputBytes = outputElementCount * sizeof(float);
        const std::vector<float> zeroOutput(outputElementCount, 0.0f);
        lastResult_.assign(outputElementCount, 0.0f);

        for (int row = 0; row < rowsM; ++row) {
            for (int col = 0; col < colsK; ++col) {
                auto vm = createVm();
                if (!vm) {
                    return false;
                }

                if (!vm->loadProgram(loadedPath_)) {
                    lastError_ = "Failed to reload the uploaded PTX program.";
                    lastResult_.clear();
                    return false;
                }

                const CUdeviceptr inputAPtr = vm->allocateMemory(inputABytes);
                const CUdeviceptr inputBPtr = vm->allocateMemory(inputBBytes);
                const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

                if (!vm->copyMemoryHtoD(inputAPtr, inputA, inputABytes)) {
                    lastError_ = "Failed to copy Matrix A into VM memory.";
                    lastResult_.clear();
                    return false;
                }

                if (!vm->copyMemoryHtoD(inputBPtr, inputB, inputBBytes)) {
                    lastError_ = "Failed to copy Matrix B into VM memory.";
                    lastResult_.clear();
                    return false;
                }

                if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
                    lastError_ = "Failed to initialize Matrix C in VM memory.";
                    lastResult_.clear();
                    return false;
                }

                std::vector<KernelParameter> params;
                params.push_back({inputAPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
                params.push_back({inputBPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
                params.push_back({outputPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
                params.push_back({
                    static_cast<CUdeviceptr>(rowsM),
                    kernel->parameters[3].size,
                    kernel->parameters[3].offset,
                });
                params.push_back({
                    static_cast<CUdeviceptr>(sharedN),
                    kernel->parameters[4].size,
                    kernel->parameters[4].offset,
                });
                params.push_back({
                    static_cast<CUdeviceptr>(colsK),
                    kernel->parameters[5].size,
                    kernel->parameters[5].offset,
                });

                vm->setKernelParameters(params);

                PTXExecutor& executor = vm->getExecutor();
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

                if (!vm->run()) {
                    lastError_ = "Kernel execution failed inside the PTX VM.";
                    lastResult_.clear();
                    return false;
                }

                const size_t outputIndex =
                    static_cast<size_t>(row) * static_cast<size_t>(colsK) + static_cast<size_t>(col);
                const CUdeviceptr outputCellPtr =
                    outputPtr + static_cast<CUdeviceptr>(outputIndex * sizeof(float));

                float outputValue = 0.0f;
                if (!vm->copyMemoryDtoH(&outputValue, outputCellPtr, sizeof(float))) {
                    lastError_ = "Kernel executed, but reading Matrix C failed.";
                    lastResult_.clear();
                    return false;
                }

                lastResult_[outputIndex] = outputValue;
            }
        }

        return true;
    }

    bool runInt8QuantizedMatMulDemo(const std::string& kernelName,
                                    const std::int8_t* inputA,
                                    int inputALen,
                                    const std::int8_t* inputB,
                                    int inputBLen,
                                    int rowsM,
                                    int colsN,
                                    int sharedK,
                                    float scaleA,
                                    float scaleB,
                                    float scaleC,
                                    int zeroPointA,
                                    int zeroPointB,
                                    int zeroPointC) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputA == nullptr || inputB == nullptr) {
            lastError_ = "Input buffers must not be null.";
            return false;
        }

        if (rowsM <= 0 || colsN <= 0 || sharedK <= 0) {
            lastError_ = "Matrix dimensions M, N, and K must all be positive integers.";
            return false;
        }

        if (!(scaleA > 0.0f && scaleB > 0.0f && scaleC > 0.0f)) {
            lastError_ = "scale_A, scale_B, and scale_C must all be positive.";
            return false;
        }

        if (zeroPointA < -128 || zeroPointA > 127 ||
            zeroPointB < -128 || zeroPointB > 127 ||
            zeroPointC < -128 || zeroPointC > 127) {
            lastError_ = "Zero-points must all be in the int8 range [-128, 127].";
            return false;
        }

        const size_t expectedALen =
            static_cast<size_t>(rowsM) * static_cast<size_t>(sharedK);
        const size_t expectedBLen =
            static_cast<size_t>(sharedK) * static_cast<size_t>(colsN);
        const size_t outputElementCount =
            static_cast<size_t>(rowsM) * static_cast<size_t>(colsN);

        if (static_cast<size_t>(inputALen) != expectedALen) {
            lastError_ = "Matrix A does not match the provided M x K dimensions.";
            return false;
        }

        if (static_cast<size_t>(inputBLen) != expectedBLen) {
            lastError_ = "Matrix B does not match the provided K x N dimensions.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isThreePointerThreeIntThreeFloatThreeIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32, .u32, .u32, .f32, .f32, .f32, .u32, .u32, .u32), such as INT8 quantized matmul.";
            return false;
        }

        constexpr unsigned int kBlockDimX = 16;
        constexpr unsigned int kBlockDimY = 16;
        const unsigned int gridDimX =
            static_cast<unsigned int>((colsN + static_cast<int>(kBlockDimX) - 1) / static_cast<int>(kBlockDimX));
        const unsigned int gridDimY =
            static_cast<unsigned int>((rowsM + static_cast<int>(kBlockDimY) - 1) / static_cast<int>(kBlockDimY));

        const size_t inputABytes = expectedALen * sizeof(std::int8_t);
        const size_t inputBBytes = expectedBLen * sizeof(std::int8_t);
        const size_t outputBytes = outputElementCount * sizeof(std::int8_t);
        const std::vector<std::int8_t> zeroOutput(outputElementCount, 0);
        lastResult_.assign(outputElementCount, 0.0f);

        for (int row = 0; row < rowsM; ++row) {
            for (int col = 0; col < colsN; ++col) {
                auto vm = createVm();
                if (!vm) {
                    return false;
                }

                if (!vm->loadProgram(loadedPath_)) {
                    lastError_ = "Failed to reload the uploaded PTX program.";
                    lastResult_.clear();
                    return false;
                }

                const CUdeviceptr inputAPtr = vm->allocateMemory(inputABytes);
                const CUdeviceptr inputBPtr = vm->allocateMemory(inputBBytes);
                const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

                if (!vm->copyMemoryHtoD(inputAPtr, inputA, inputABytes)) {
                    lastError_ = "Failed to copy Matrix A into VM memory.";
                    lastResult_.clear();
                    return false;
                }

                if (!vm->copyMemoryHtoD(inputBPtr, inputB, inputBBytes)) {
                    lastError_ = "Failed to copy Matrix B into VM memory.";
                    lastResult_.clear();
                    return false;
                }

                if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
                    lastError_ = "Failed to initialize Matrix C in VM memory.";
                    lastResult_.clear();
                    return false;
                }

                std::vector<KernelParameter> params;
                params.push_back({inputAPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
                params.push_back({inputBPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
                params.push_back({outputPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
                params.push_back({
                    static_cast<CUdeviceptr>(rowsM),
                    kernel->parameters[3].size,
                    kernel->parameters[3].offset,
                });
                params.push_back({
                    static_cast<CUdeviceptr>(colsN),
                    kernel->parameters[4].size,
                    kernel->parameters[4].offset,
                });
                params.push_back({
                    static_cast<CUdeviceptr>(sharedK),
                    kernel->parameters[5].size,
                    kernel->parameters[5].offset,
                });
                params.push_back({packFloat32(scaleA), kernel->parameters[6].size, kernel->parameters[6].offset});
                params.push_back({packFloat32(scaleB), kernel->parameters[7].size, kernel->parameters[7].offset});
                params.push_back({packFloat32(scaleC), kernel->parameters[8].size, kernel->parameters[8].offset});
                params.push_back({
                    static_cast<CUdeviceptr>(static_cast<std::uint32_t>(zeroPointA)),
                    kernel->parameters[9].size,
                    kernel->parameters[9].offset,
                });
                params.push_back({
                    static_cast<CUdeviceptr>(static_cast<std::uint32_t>(zeroPointB)),
                    kernel->parameters[10].size,
                    kernel->parameters[10].offset,
                });
                params.push_back({
                    static_cast<CUdeviceptr>(static_cast<std::uint32_t>(zeroPointC)),
                    kernel->parameters[11].size,
                    kernel->parameters[11].offset,
                });

                vm->setKernelParameters(params);

                PTXExecutor& executor = vm->getExecutor();
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

                if (!vm->run()) {
                    lastError_ = "Kernel execution failed inside the PTX VM.";
                    lastResult_.clear();
                    return false;
                }

                const size_t outputIndex =
                    static_cast<size_t>(row) * static_cast<size_t>(colsN) + static_cast<size_t>(col);
                const CUdeviceptr outputCellPtr =
                    outputPtr + static_cast<CUdeviceptr>(outputIndex * sizeof(std::int8_t));

                std::int8_t outputValue = 0;
                if (!vm->copyMemoryDtoH(&outputValue, outputCellPtr, sizeof(std::int8_t))) {
                    lastError_ = "Kernel executed, but reading Matrix C failed.";
                    lastResult_.clear();
                    return false;
                }

                lastResult_[outputIndex] = static_cast<float>(outputValue);
            }
        }

        return true;
    }

    bool runSparseMatVecDemo(const std::string& kernelName,
                             const float* inputA,
                             int inputALen,
                             const float* inputB,
                             int inputBLen,
                             int rowsM,
                             int colsN,
                             int nnz) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputA == nullptr || inputB == nullptr) {
            lastError_ = "Input buffers must not be null.";
            return false;
        }

        if (rowsM <= 0 || colsN <= 0 || nnz < 0) {
            lastError_ = "M and N must be positive integers, and nnz must be non-negative.";
            return false;
        }

        const size_t expectedALen =
            static_cast<size_t>(rowsM) * static_cast<size_t>(colsN);
        const size_t expectedBLen = static_cast<size_t>(colsN);
        const size_t outputLen = static_cast<size_t>(rowsM);

        if (static_cast<size_t>(inputALen) != expectedALen) {
            lastError_ = "Matrix A does not match the provided M x N dimensions.";
            return false;
        }

        if (static_cast<size_t>(inputBLen) != expectedBLen) {
            lastError_ = "Vector x does not match the provided N dimension.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isMatrixMultiplySignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32, .u32, .u32), such as sparse matrix-vector multiplication.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputABytes = expectedALen * sizeof(float);
        const size_t inputBBytes = expectedBLen * sizeof(float);
        const size_t outputBytes = outputLen * sizeof(float);
        const std::vector<float> zeroOutput(outputLen, 0.0f);
        const CUdeviceptr inputAPtr = vm->allocateMemory(inputABytes);
        const CUdeviceptr inputBPtr = vm->allocateMemory(inputBBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputAPtr, inputA, inputABytes)) {
            lastError_ = "Failed to copy Matrix A into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputBPtr, inputB, inputBBytes)) {
            lastError_ = "Failed to copy vector x into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output vector in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputAPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({inputBPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({outputPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({static_cast<CUdeviceptr>(rowsM), kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(colsN), kernel->parameters[4].size, kernel->parameters[4].offset});
        params.push_back({static_cast<CUdeviceptr>(nnz), kernel->parameters[5].size, kernel->parameters[5].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(outputLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the sparse matvec output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runBatchedMatrixMultiplicationDemo(const std::string& kernelName,
                                            const float* inputA,
                                            int inputALen,
                                            const float* inputB,
                                            int inputBLen,
                                            int batchSize,
                                            int rowsM,
                                            int sharedN,
                                            int colsK) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputA == nullptr || inputB == nullptr) {
            lastError_ = "Input buffers must not be null.";
            return false;
        }

        if (batchSize <= 0 || rowsM <= 0 || sharedN <= 0 || colsK <= 0) {
            lastError_ = "BATCH, M, N, and K must all be positive integers.";
            return false;
        }

        const size_t expectedALen =
            static_cast<size_t>(batchSize) * static_cast<size_t>(rowsM) * static_cast<size_t>(sharedN);
        const size_t expectedBLen =
            static_cast<size_t>(batchSize) * static_cast<size_t>(sharedN) * static_cast<size_t>(colsK);
        const size_t outputLen =
            static_cast<size_t>(batchSize) * static_cast<size_t>(rowsM) * static_cast<size_t>(colsK);

        if (static_cast<size_t>(inputALen) != expectedALen) {
            lastError_ = "Tensor A does not match the provided BATCH x M x N dimensions.";
            return false;
        }

        if (static_cast<size_t>(inputBLen) != expectedBLen) {
            lastError_ = "Tensor B does not match the provided BATCH x N x K dimensions.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isThreePointerFourIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32, .u32, .u32, .u32), such as batched matrix multiplication.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputABytes = expectedALen * sizeof(float);
        const size_t inputBBytes = expectedBLen * sizeof(float);
        const size_t outputBytes = outputLen * sizeof(float);
        const std::vector<float> zeroOutput(outputLen, 0.0f);
        const CUdeviceptr inputAPtr = vm->allocateMemory(inputABytes);
        const CUdeviceptr inputBPtr = vm->allocateMemory(inputBBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputAPtr, inputA, inputABytes)) {
            lastError_ = "Failed to copy Tensor A into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputBPtr, inputB, inputBBytes)) {
            lastError_ = "Failed to copy Tensor B into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize Tensor C in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputAPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({inputBPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({outputPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({static_cast<CUdeviceptr>(batchSize), kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(rowsM), kernel->parameters[4].size, kernel->parameters[4].offset});
        params.push_back({static_cast<CUdeviceptr>(sharedN), kernel->parameters[5].size, kernel->parameters[5].offset});
        params.push_back({static_cast<CUdeviceptr>(colsK), kernel->parameters[6].size, kernel->parameters[6].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(outputLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the batched matmul output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runMultiAgentSimulationDemo(const std::string& kernelName,
                                     const float* input,
                                     int inputLen,
                                     int agentCount) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (agentCount <= 0) {
            lastError_ = "N must be a positive integer.";
            return false;
        }

        const size_t expectedLen = static_cast<size_t>(agentCount) * 4u;
        if (static_cast<size_t>(inputLen) != expectedLen) {
            lastError_ = "Agent buffer length must equal 4 x N.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isReductionSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u32), such as multi-agent simulation.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedLen * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(expectedLen, 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the agent buffer into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output agent buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({
            static_cast<CUdeviceptr>(agentCount),
            kernel->parameters[2].size,
            kernel->parameters[2].offset,
        });
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(expectedLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the output agent buffer failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runNearestNeighborDemo(const std::string& kernelName,
                                const float* input,
                                int inputLen,
                                int pointCount) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (pointCount <= 0) {
            lastError_ = "N must be a positive integer.";
            return false;
        }

        const size_t expectedInputLen = static_cast<size_t>(pointCount) * 3u;
        if (static_cast<size_t>(inputLen) != expectedInputLen) {
            lastError_ = "Point buffer length must equal 3 x N.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isReductionSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u32), such as nearest neighbor.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedInputLen * sizeof(float);
        const size_t outputBytes = static_cast<size_t>(pointCount) * sizeof(std::int32_t);
        const std::vector<std::int32_t> initialOutput(static_cast<size_t>(pointCount), -1);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the point cloud into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, initialOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output index buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({
            static_cast<CUdeviceptr>(pointCount),
            kernel->parameters[2].size,
            kernel->parameters[2].offset,
        });
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        std::vector<std::int32_t> output(static_cast<size_t>(pointCount), -1);
        if (!vm->copyMemoryDtoH(output.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the output index buffer failed.";
            return false;
        }

        lastResult_.clear();
        lastResult_.reserve(static_cast<size_t>(pointCount));
        for (std::int32_t value : output) {
            lastResult_.push_back(static_cast<float>(value));
        }

        return true;
    }

    bool runKMeansDemo(const std::string& kernelName,
                       const float* dataX,
                       const float* dataY,
                       const float* initialCentroidX,
                       const float* initialCentroidY,
                       int sampleSize,
                       int clusterCount,
                       int maxIterations) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (dataX == nullptr || dataY == nullptr || initialCentroidX == nullptr || initialCentroidY == nullptr) {
            lastError_ = "Input buffers must not be null.";
            return false;
        }

        if (sampleSize <= 0 || clusterCount <= 0 || maxIterations <= 0) {
            lastError_ = "sample_size, k, and max_iterations must all be positive integers.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isSevenPointerThreeIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u64, .u64, .u64, .u64, .u32, .u32, .u32), such as k-means.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t sampleBytes = static_cast<size_t>(sampleSize) * sizeof(float);
        const size_t labelBytes = static_cast<size_t>(sampleSize) * sizeof(std::int32_t);
        const size_t centroidBytes = static_cast<size_t>(clusterCount) * sizeof(float);
        const std::vector<std::int32_t> initialLabels(static_cast<size_t>(sampleSize), -1);
        const CUdeviceptr dataXPtr = vm->allocateMemory(sampleBytes);
        const CUdeviceptr dataYPtr = vm->allocateMemory(sampleBytes);
        const CUdeviceptr labelsPtr = vm->allocateMemory(labelBytes);
        const CUdeviceptr initialCentroidXPtr = vm->allocateMemory(centroidBytes);
        const CUdeviceptr initialCentroidYPtr = vm->allocateMemory(centroidBytes);
        const CUdeviceptr finalCentroidXPtr = vm->allocateMemory(centroidBytes);
        const CUdeviceptr finalCentroidYPtr = vm->allocateMemory(centroidBytes);

        if (!vm->copyMemoryHtoD(dataXPtr, dataX, sampleBytes) ||
            !vm->copyMemoryHtoD(dataYPtr, dataY, sampleBytes)) {
            lastError_ = "Failed to copy point coordinates into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(labelsPtr, initialLabels.data(), labelBytes)) {
            lastError_ = "Failed to initialize the labels buffer in VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(initialCentroidXPtr, initialCentroidX, centroidBytes) ||
            !vm->copyMemoryHtoD(initialCentroidYPtr, initialCentroidY, centroidBytes) ||
            !vm->copyMemoryHtoD(finalCentroidXPtr, initialCentroidX, centroidBytes) ||
            !vm->copyMemoryHtoD(finalCentroidYPtr, initialCentroidY, centroidBytes)) {
            lastError_ = "Failed to initialize centroid buffers in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({dataXPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({dataYPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({labelsPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({initialCentroidXPtr, kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({initialCentroidYPtr, kernel->parameters[4].size, kernel->parameters[4].offset});
        params.push_back({finalCentroidXPtr, kernel->parameters[5].size, kernel->parameters[5].offset});
        params.push_back({finalCentroidYPtr, kernel->parameters[6].size, kernel->parameters[6].offset});
        params.push_back({static_cast<CUdeviceptr>(sampleSize), kernel->parameters[7].size, kernel->parameters[7].offset});
        params.push_back({static_cast<CUdeviceptr>(clusterCount), kernel->parameters[8].size, kernel->parameters[8].offset});
        params.push_back({static_cast<CUdeviceptr>(maxIterations), kernel->parameters[9].size, kernel->parameters[9].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        std::vector<std::int32_t> labels(static_cast<size_t>(sampleSize), -1);
        std::vector<float> finalCentroidX(static_cast<size_t>(clusterCount), 0.0f);
        std::vector<float> finalCentroidY(static_cast<size_t>(clusterCount), 0.0f);
        if (!vm->copyMemoryDtoH(labels.data(), labelsPtr, labelBytes) ||
            !vm->copyMemoryDtoH(finalCentroidX.data(), finalCentroidXPtr, centroidBytes) ||
            !vm->copyMemoryDtoH(finalCentroidY.data(), finalCentroidYPtr, centroidBytes)) {
            lastError_ = "Kernel executed, but reading the k-means outputs failed.";
            return false;
        }

        lastResult_.clear();
        lastResult_.reserve(static_cast<size_t>(sampleSize + clusterCount * 2));
        for (std::int32_t value : labels) {
            lastResult_.push_back(static_cast<float>(value));
        }
        lastResult_.insert(lastResult_.end(), finalCentroidX.begin(), finalCentroidX.end());
        lastResult_.insert(lastResult_.end(), finalCentroidY.begin(), finalCentroidY.end());
        return true;
    }

    bool runBatchNormalizationDemo(const std::string& kernelName,
                                   const float* input,
                                   int inputLen,
                                   const float* gamma,
                                   int gammaLen,
                                   const float* beta,
                                   int betaLen,
                                   int rowsN,
                                   int colsC,
                                   float eps) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr || gamma == nullptr || beta == nullptr) {
            lastError_ = "Input buffers must not be null.";
            return false;
        }

        if (rowsN <= 0 || colsC <= 0) {
            lastError_ = "N and C must both be positive integers.";
            return false;
        }

        const size_t expectedInputLen = static_cast<size_t>(rowsN) * static_cast<size_t>(colsC);
        if (static_cast<size_t>(inputLen) != expectedInputLen) {
            lastError_ = "Input tensor length must equal N x C.";
            return false;
        }

        if (gammaLen != colsC || betaLen != colsC) {
            lastError_ = "Gamma and beta lengths must both equal C.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isFourPointerTwoIntOneFloatSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u64, .u32, .u32, .f32), such as batch normalization.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedInputLen * sizeof(float);
        const size_t paramBytes = static_cast<size_t>(colsC) * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(expectedInputLen, 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr gammaPtr = vm->allocateMemory(paramBytes);
        const CUdeviceptr betaPtr = vm->allocateMemory(paramBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input tensor into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(gammaPtr, gamma, paramBytes)) {
            lastError_ = "Failed to copy gamma into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(betaPtr, beta, paramBytes)) {
            lastError_ = "Failed to copy beta into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output tensor in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({gammaPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({betaPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({outputPtr, kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(rowsN), kernel->parameters[4].size, kernel->parameters[4].offset});
        params.push_back({static_cast<CUdeviceptr>(colsC), kernel->parameters[5].size, kernel->parameters[5].offset});

        std::uint32_t epsBits = 0;
        std::memcpy(&epsBits, &eps, sizeof(float));
        params.push_back({static_cast<CUdeviceptr>(epsBits), kernel->parameters[6].size, kernel->parameters[6].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(expectedInputLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the batch normalization output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runMaxPooling2DDemo(const std::string& kernelName,
                             const float* input,
                             int inputLen,
                             int batchSize,
                             int channelCount,
                             int height,
                             int width,
                             int kernelSize,
                             int stride,
                             int padding) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (batchSize <= 0 || channelCount <= 0 || height <= 0 || width <= 0 ||
            kernelSize <= 0 || stride <= 0 || padding < 0) {
            lastError_ = "N, C, H, W, kernel_size, and stride must be positive, and padding must be non-negative.";
            return false;
        }

        const int outputHeight = (height + 2 * padding - kernelSize) / stride + 1;
        const int outputWidth = (width + 2 * padding - kernelSize) / stride + 1;
        if (outputHeight <= 0 || outputWidth <= 0) {
            lastError_ = "The provided pooling parameters produce an empty output tensor.";
            return false;
        }

        const size_t expectedInputLen =
            static_cast<size_t>(batchSize) * static_cast<size_t>(channelCount) *
            static_cast<size_t>(height) * static_cast<size_t>(width);
        const size_t outputLen =
            static_cast<size_t>(batchSize) * static_cast<size_t>(channelCount) *
            static_cast<size_t>(outputHeight) * static_cast<size_t>(outputWidth);

        if (static_cast<size_t>(inputLen) != expectedInputLen) {
            lastError_ = "Input tensor length must equal N x C x H x W.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isTwoPointerSevenIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u32, .u32, .u32, .u32, .u32, .u32, .u32), such as 2D max pooling.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedInputLen * sizeof(float);
        const size_t outputBytes = outputLen * sizeof(float);
        const std::vector<float> zeroOutput(outputLen, 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input tensor into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output tensor in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({static_cast<CUdeviceptr>(batchSize), kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({static_cast<CUdeviceptr>(channelCount), kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(height), kernel->parameters[4].size, kernel->parameters[4].offset});
        params.push_back({static_cast<CUdeviceptr>(width), kernel->parameters[5].size, kernel->parameters[5].offset});
        params.push_back({static_cast<CUdeviceptr>(kernelSize), kernel->parameters[6].size, kernel->parameters[6].offset});
        params.push_back({static_cast<CUdeviceptr>(stride), kernel->parameters[7].size, kernel->parameters[7].offset});
        params.push_back({static_cast<CUdeviceptr>(padding), kernel->parameters[8].size, kernel->parameters[8].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(outputLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the max-pooling output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runRegressionVectorDemo(const std::string& kernelName,
                                 const float* inputA,
                                 int inputALen,
                                 const float* inputB,
                                 int inputBLen,
                                 int sampleCount,
                                 int featureCount) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputA == nullptr || inputB == nullptr) {
            lastError_ = "Input buffers must not be null.";
            return false;
        }

        if (sampleCount <= 0 || featureCount <= 0) {
            lastError_ = "n_samples and n_features must both be positive integers.";
            return false;
        }

        const size_t expectedALen =
            static_cast<size_t>(sampleCount) * static_cast<size_t>(featureCount);
        const size_t expectedBLen = static_cast<size_t>(sampleCount);
        const size_t outputLen = static_cast<size_t>(featureCount);

        if (static_cast<size_t>(inputALen) != expectedALen) {
            lastError_ = "Feature matrix length must equal n_samples x n_features.";
            return false;
        }

        if (static_cast<size_t>(inputBLen) != expectedBLen) {
            lastError_ = "Target vector length must equal n_samples.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isThreePointerTwoIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32, .u32), such as regression coefficient output.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputABytes = expectedALen * sizeof(float);
        const size_t inputBBytes = expectedBLen * sizeof(float);
        const size_t outputBytes = outputLen * sizeof(float);
        const std::vector<float> zeroOutput(outputLen, 0.0f);
        const CUdeviceptr inputAPtr = vm->allocateMemory(inputABytes);
        const CUdeviceptr inputBPtr = vm->allocateMemory(inputBBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputAPtr, inputA, inputABytes)) {
            lastError_ = "Failed to copy the feature matrix into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputBPtr, inputB, inputBBytes)) {
            lastError_ = "Failed to copy the target vector into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the coefficient buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputAPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({inputBPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({outputPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({static_cast<CUdeviceptr>(sampleCount), kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(featureCount), kernel->parameters[4].size, kernel->parameters[4].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(outputLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the coefficient output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runFftDemo(const std::string& kernelName,
                    const float* input,
                    int inputLen,
                    int signalLength) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (signalLength <= 0) {
            lastError_ = "N must be a positive integer.";
            return false;
        }

        const size_t expectedLen = static_cast<size_t>(signalLength) * 2u;
        if (static_cast<size_t>(inputLen) != expectedLen) {
            lastError_ = "Signal length must equal 2 x N for interleaved complex input.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isReductionSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u32), such as FFT output buffers.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedLen * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(expectedLen, 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the complex signal into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the spectrum buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({
            static_cast<CUdeviceptr>(signalLength),
            kernel->parameters[2].size,
            kernel->parameters[2].offset,
        });
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(expectedLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the FFT spectrum failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runMatrixAdditionDemo(const std::string& kernelName,
                               const float* inputA,
                               int inputALen,
                               const float* inputB,
                               int inputBLen,
                               int matrixN) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputA == nullptr || inputB == nullptr) {
            lastError_ = "Input buffers must not be null.";
            return false;
        }

        if (matrixN <= 0) {
            lastError_ = "Matrix N must be a positive integer.";
            return false;
        }

        const size_t expectedLen =
            static_cast<size_t>(matrixN) * static_cast<size_t>(matrixN);
        if (static_cast<size_t>(inputALen) != expectedLen) {
            lastError_ = "Matrix A does not match the provided N x N dimensions.";
            return false;
        }

        if (static_cast<size_t>(inputBLen) != expectedLen) {
            lastError_ = "Matrix B does not match the provided N x N dimensions.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isMatrixAdditionSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32), such as matrix addition.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedLen * sizeof(float);
        const size_t outputBytes = expectedLen * sizeof(float);
        const std::vector<float> zeroOutput(expectedLen, 0.0f);
        const CUdeviceptr inputAPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr inputBPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputAPtr, inputA, inputBytes)) {
            lastError_ = "Failed to copy Matrix A into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputBPtr, inputB, inputBytes)) {
            lastError_ = "Failed to copy Matrix B into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize Matrix C in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputAPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({inputBPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({outputPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({
            static_cast<CUdeviceptr>(matrixN),
            kernel->parameters[3].size,
            kernel->parameters[3].offset,
        });

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.resize(expectedLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading Matrix C failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runConvolution1DDemo(const std::string& kernelName,
                              const float* input,
                              int inputLen,
                              const float* kernelInput,
                              int kernelLen) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr || kernelInput == nullptr) {
            lastError_ = "Input and kernel buffers must not be null.";
            return false;
        }

        if (inputLen <= 0 || kernelLen <= 0) {
            lastError_ = "Input and kernel arrays must both contain at least one float.";
            return false;
        }

        if (kernelLen > inputLen) {
            lastError_ = "Kernel size must not exceed the input size.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isConvolution1DSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32, .u32), such as 1D convolution.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputLen) * sizeof(float);
        const size_t kernelBytes = static_cast<size_t>(kernelLen) * sizeof(float);
        const size_t outputLen =
            static_cast<size_t>(inputLen - kernelLen + 1);
        const size_t outputBytes = outputLen * sizeof(float);
        const std::vector<float> zeroOutput(outputLen, 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr kernelPtr = vm->allocateMemory(kernelBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input array into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(kernelPtr, kernelInput, kernelBytes)) {
            lastError_ = "Failed to copy the kernel array into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output array in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({kernelPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({outputPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({
            static_cast<CUdeviceptr>(inputLen),
            kernel->parameters[3].size,
            kernel->parameters[3].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(kernelLen),
            kernel->parameters[4].size,
            kernel->parameters[4].offset,
        });

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.resize(outputLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the convolution output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runConvolution2DDemo(const std::string& kernelName,
                              const float* input,
                              int inputLen,
                              const float* kernelInput,
                              int kernelLen,
                              int inputRows,
                              int inputCols,
                              int kernelRows,
                              int kernelCols) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr || kernelInput == nullptr) {
            lastError_ = "Input and kernel buffers must not be null.";
            return false;
        }

        if (inputRows <= 0 || inputCols <= 0 || kernelRows <= 0 || kernelCols <= 0) {
            lastError_ = "Input and kernel dimensions must all be positive integers.";
            return false;
        }

        if (kernelRows > inputRows || kernelCols > inputCols) {
            lastError_ = "Kernel dimensions must not exceed the input dimensions.";
            return false;
        }

        const size_t expectedInputLen =
            static_cast<size_t>(inputRows) * static_cast<size_t>(inputCols);
        const size_t expectedKernelLen =
            static_cast<size_t>(kernelRows) * static_cast<size_t>(kernelCols);
        const int outputRows = inputRows - kernelRows + 1;
        const int outputCols = inputCols - kernelCols + 1;
        const size_t outputLen =
            static_cast<size_t>(outputRows) * static_cast<size_t>(outputCols);

        if (static_cast<size_t>(inputLen) != expectedInputLen) {
            lastError_ = "Input matrix length does not match input_rows x input_cols.";
            return false;
        }

        if (static_cast<size_t>(kernelLen) != expectedKernelLen) {
            lastError_ = "Kernel length does not match kernel_rows x kernel_cols.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isConvolution2DSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32, .u32, .u32, .u32), such as 2D convolution.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedInputLen * sizeof(float);
        const size_t kernelBytes = expectedKernelLen * sizeof(float);
        const size_t outputBytes = outputLen * sizeof(float);
        const std::vector<float> zeroOutput(outputLen, 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr kernelPtr = vm->allocateMemory(kernelBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input matrix into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(kernelPtr, kernelInput, kernelBytes)) {
            lastError_ = "Failed to copy the kernel matrix into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output matrix in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({kernelPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({outputPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({
            static_cast<CUdeviceptr>(inputRows),
            kernel->parameters[3].size,
            kernel->parameters[3].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(inputCols),
            kernel->parameters[4].size,
            kernel->parameters[4].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(kernelRows),
            kernel->parameters[5].size,
            kernel->parameters[5].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(kernelCols),
            kernel->parameters[6].size,
            kernel->parameters[6].offset,
        });

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.resize(outputLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the 2D convolution output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runPaddedConvolution2DDemo(const std::string& kernelName,
                                    const float* input,
                                    int inputLen,
                                    const float* kernelInput,
                                    int kernelLen,
                                    int inputRows,
                                    int inputCols,
                                    int kernelRows,
                                    int kernelCols) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr || kernelInput == nullptr) {
            lastError_ = "Input and kernel buffers must not be null.";
            return false;
        }

        if (inputRows <= 0 || inputCols <= 0 || kernelRows <= 0 || kernelCols <= 0) {
            lastError_ = "Input and kernel dimensions must all be positive integers.";
            return false;
        }

        const size_t expectedInputLen =
            static_cast<size_t>(inputRows) * static_cast<size_t>(inputCols);
        const size_t expectedKernelLen =
            static_cast<size_t>(kernelRows) * static_cast<size_t>(kernelCols);
        const size_t outputLen = expectedInputLen;

        if (static_cast<size_t>(inputLen) != expectedInputLen) {
            lastError_ = "Input matrix length does not match input_rows x input_cols.";
            return false;
        }

        if (static_cast<size_t>(kernelLen) != expectedKernelLen) {
            lastError_ = "Kernel length does not match kernel_rows x kernel_cols.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isConvolution2DSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32, .u32, .u32, .u32), such as Gaussian blur.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedInputLen * sizeof(float);
        const size_t kernelBytes = expectedKernelLen * sizeof(float);
        const size_t outputBytes = outputLen * sizeof(float);
        const std::vector<float> zeroOutput(outputLen, 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr kernelPtr = vm->allocateMemory(kernelBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input matrix into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(kernelPtr, kernelInput, kernelBytes)) {
            lastError_ = "Failed to copy the kernel matrix into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output matrix in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({kernelPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({outputPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({
            static_cast<CUdeviceptr>(inputRows),
            kernel->parameters[3].size,
            kernel->parameters[3].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(inputCols),
            kernel->parameters[4].size,
            kernel->parameters[4].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(kernelRows),
            kernel->parameters[5].size,
            kernel->parameters[5].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(kernelCols),
            kernel->parameters[6].size,
            kernel->parameters[6].offset,
        });

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.resize(outputLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the Gaussian blur output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runConvolution3DDemo(const std::string& kernelName,
                              const float* input,
                              int inputLen,
                              const float* kernelInput,
                              int kernelLen,
                              int inputDepth,
                              int inputRows,
                              int inputCols,
                              int kernelDepth,
                              int kernelRows,
                              int kernelCols) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr || kernelInput == nullptr) {
            lastError_ = "Input and kernel buffers must not be null.";
            return false;
        }

        if (inputDepth <= 0 || inputRows <= 0 || inputCols <= 0 ||
            kernelDepth <= 0 || kernelRows <= 0 || kernelCols <= 0) {
            lastError_ = "Input and kernel dimensions must all be positive integers.";
            return false;
        }

        if (kernelDepth > inputDepth || kernelRows > inputRows || kernelCols > inputCols) {
            lastError_ = "Kernel dimensions must not exceed the input dimensions.";
            return false;
        }

        const size_t expectedInputLen =
            static_cast<size_t>(inputDepth) *
            static_cast<size_t>(inputRows) *
            static_cast<size_t>(inputCols);
        const size_t expectedKernelLen =
            static_cast<size_t>(kernelDepth) *
            static_cast<size_t>(kernelRows) *
            static_cast<size_t>(kernelCols);
        const int outputDepth = inputDepth - kernelDepth + 1;
        const int outputRows = inputRows - kernelRows + 1;
        const int outputCols = inputCols - kernelCols + 1;
        const size_t outputLen =
            static_cast<size_t>(outputDepth) *
            static_cast<size_t>(outputRows) *
            static_cast<size_t>(outputCols);

        if (static_cast<size_t>(inputLen) != expectedInputLen) {
            lastError_ = "Input volume length does not match input_depth x input_rows x input_cols.";
            return false;
        }

        if (static_cast<size_t>(kernelLen) != expectedKernelLen) {
            lastError_ = "Kernel volume length does not match kernel_depth x kernel_rows x kernel_cols.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isConvolution3DSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32, .u32, .u32, .u32, .u32, .u32), such as 3D convolution.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedInputLen * sizeof(float);
        const size_t kernelBytes = expectedKernelLen * sizeof(float);
        const size_t outputBytes = outputLen * sizeof(float);
        const std::vector<float> zeroOutput(outputLen, 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr kernelPtr = vm->allocateMemory(kernelBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input volume into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(kernelPtr, kernelInput, kernelBytes)) {
            lastError_ = "Failed to copy the kernel volume into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output volume in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({kernelPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({outputPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({
            static_cast<CUdeviceptr>(inputDepth),
            kernel->parameters[3].size,
            kernel->parameters[3].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(inputRows),
            kernel->parameters[4].size,
            kernel->parameters[4].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(inputCols),
            kernel->parameters[5].size,
            kernel->parameters[5].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(kernelDepth),
            kernel->parameters[6].size,
            kernel->parameters[6].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(kernelRows),
            kernel->parameters[7].size,
            kernel->parameters[7].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(kernelCols),
            kernel->parameters[8].size,
            kernel->parameters[8].offset,
        });

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.resize(outputLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the 3D convolution output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runMatrixTransposeDemo(const std::string& kernelName,
                                const float* inputA,
                                int inputALen,
                                int rows,
                                int cols) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputA == nullptr) {
            lastError_ = "Input matrix buffer must not be null.";
            return false;
        }

        if (rows <= 0 || cols <= 0) {
            lastError_ = "Input matrix rows and cols must both be positive integers.";
            return false;
        }

        const size_t expectedInputLen =
            static_cast<size_t>(rows) * static_cast<size_t>(cols);
        if (static_cast<size_t>(inputALen) != expectedInputLen) {
            lastError_ = "Input matrix length does not match the provided rows x cols dimensions.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isMatrixTransposeSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u32, .u32), such as matrix transpose.";
            return false;
        }

        constexpr unsigned int kBlockDimX = 16;
        constexpr unsigned int kBlockDimY = 16;
        const unsigned int gridDimX =
            static_cast<unsigned int>((cols + static_cast<int>(kBlockDimX) - 1) / static_cast<int>(kBlockDimX));
        const unsigned int gridDimY =
            static_cast<unsigned int>((rows + static_cast<int>(kBlockDimY) - 1) / static_cast<int>(kBlockDimY));

        const size_t inputBytes = expectedInputLen * sizeof(float);
        const size_t outputElementCount = expectedInputLen;
        const size_t outputBytes = outputElementCount * sizeof(float);
        const std::vector<float> zeroOutput(outputElementCount, 0.0f);
        lastResult_.assign(outputElementCount, 0.0f);

        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                auto vm = createVm();
                if (!vm) {
                    return false;
                }

                if (!vm->loadProgram(loadedPath_)) {
                    lastError_ = "Failed to reload the uploaded PTX program.";
                    lastResult_.clear();
                    return false;
                }

                const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
                const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

                if (!vm->copyMemoryHtoD(inputPtr, inputA, inputBytes)) {
                    lastError_ = "Failed to copy the input matrix into VM memory.";
                    lastResult_.clear();
                    return false;
                }

                if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
                    lastError_ = "Failed to initialize the output matrix in VM memory.";
                    lastResult_.clear();
                    return false;
                }

                std::vector<KernelParameter> params;
                params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
                params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
                params.push_back({
                    static_cast<CUdeviceptr>(rows),
                    kernel->parameters[2].size,
                    kernel->parameters[2].offset,
                });
                params.push_back({
                    static_cast<CUdeviceptr>(cols),
                    kernel->parameters[3].size,
                    kernel->parameters[3].offset,
                });

                vm->setKernelParameters(params);

                PTXExecutor& executor = vm->getExecutor();
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

                if (!vm->run()) {
                    lastError_ = "Kernel execution failed inside the PTX VM.";
                    lastResult_.clear();
                    return false;
                }

                const size_t outputIndex =
                    static_cast<size_t>(col) * static_cast<size_t>(rows) + static_cast<size_t>(row);
                const CUdeviceptr outputCellPtr =
                    outputPtr + static_cast<CUdeviceptr>(outputIndex * sizeof(float));

                float outputValue = 0.0f;
                if (!vm->copyMemoryDtoH(&outputValue, outputCellPtr, sizeof(float))) {
                    lastError_ = "Kernel executed, but reading the transposed output failed.";
                    lastResult_.clear();
                    return false;
                }

                lastResult_[outputIndex] = outputValue;
            }
        }

        return true;
    }

    bool runReductionDemo(const std::string& kernelName,
                          const float* input,
                          int inputLen) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (inputLen <= 0) {
            lastError_ = "Input array must contain at least one float.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isReductionSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u32), such as reduction.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputLen) * sizeof(float);
        const size_t outputBytes = sizeof(float);
        const float zeroOutput = 0.0f;
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input array into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, &zeroOutput, outputBytes)) {
            lastError_ = "Failed to initialize the output buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({
            static_cast<CUdeviceptr>(inputLen),
            kernel->parameters[2].size,
            kernel->parameters[2].offset,
        });

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        float outputValue = 0.0f;
        if (!vm->copyMemoryDtoH(&outputValue, outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the reduced output failed.";
            return false;
        }

        lastResult_.assign(1, outputValue);
        return true;
    }

    bool runHistogrammingDemo(const std::string& kernelName,
                              const std::int32_t* input,
                              int inputLen,
                              int numBins) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (inputLen <= 0 || numBins <= 0) {
            lastError_ = "N and num_bins must both be positive integers.";
            return false;
        }

        for (int i = 0; i < inputLen; ++i) {
            if (input[i] < 0 || input[i] >= numBins) {
                lastError_ = "Every input value must be in the range [0, num_bins).";
                return false;
            }
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isHistogrammingSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u32, .u32), such as histogramming.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputLen) * sizeof(std::int32_t);
        const size_t outputBytes = static_cast<size_t>(numBins) * sizeof(std::int32_t);
        const std::vector<std::int32_t> zeroHistogram(static_cast<size_t>(numBins), 0);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input array into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroHistogram.data(), outputBytes)) {
            lastError_ = "Failed to initialize the histogram buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({
            static_cast<CUdeviceptr>(inputLen),
            kernel->parameters[2].size,
            kernel->parameters[2].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(numBins),
            kernel->parameters[3].size,
            kernel->parameters[3].offset,
        });

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        std::vector<std::int32_t> histogram(static_cast<size_t>(numBins), 0);
        if (!vm->copyMemoryDtoH(histogram.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the histogram output failed.";
            return false;
        }

        lastResult_.assign(static_cast<size_t>(numBins), 0.0f);
        for (int i = 0; i < numBins; ++i) {
            lastResult_[static_cast<size_t>(i)] = static_cast<float>(histogram[static_cast<size_t>(i)]);
        }

        return true;
    }

    bool runCrossEntropyDemo(const std::string& kernelName,
                             const float* logits,
                             int logitsLen,
                             const std::int32_t* labels,
                             int labelsLen,
                             int rowsN,
                             int colsC) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (logits == nullptr || labels == nullptr) {
            lastError_ = "Logits and label buffers must not be null.";
            return false;
        }

        if (rowsN <= 0 || colsC <= 0) {
            lastError_ = "N and C must both be positive integers.";
            return false;
        }

        const size_t expectedLogitsLen =
            static_cast<size_t>(rowsN) * static_cast<size_t>(colsC);
        if (static_cast<size_t>(logitsLen) != expectedLogitsLen) {
            lastError_ = "Logits length does not match N x C.";
            return false;
        }

        if (labelsLen != rowsN) {
            lastError_ = "Label count must match N.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isThreePointerTwoIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32, .u32), such as categorical cross entropy.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t logitsBytes = expectedLogitsLen * sizeof(float);
        const size_t labelBytes = static_cast<size_t>(rowsN) * sizeof(std::int32_t);
        const size_t outputBytes = sizeof(float);
        const float zeroOutput = 0.0f;
        const CUdeviceptr logitsPtr = vm->allocateMemory(logitsBytes);
        const CUdeviceptr labelsPtr = vm->allocateMemory(labelBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(logitsPtr, logits, logitsBytes)) {
            lastError_ = "Failed to copy logits into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(labelsPtr, labels, labelBytes)) {
            lastError_ = "Failed to copy labels into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, &zeroOutput, outputBytes)) {
            lastError_ = "Failed to initialize the loss output in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({logitsPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({labelsPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({outputPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({static_cast<CUdeviceptr>(rowsN), kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(colsC), kernel->parameters[4].size, kernel->parameters[4].offset});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        float outputValue = 0.0f;
        if (!vm->copyMemoryDtoH(&outputValue, outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the cross entropy output failed.";
            return false;
        }

        lastResult_.assign(1, outputValue);
        return true;
    }

    bool runMonteCarloDemo(const std::string& kernelName,
                           const float* samples,
                           int sampleCount,
                           float lowerBound,
                           float upperBound) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (samples == nullptr) {
            lastError_ = "Sample buffer must not be null.";
            return false;
        }

        if (sampleCount <= 0) {
            lastError_ = "Sample buffer must contain at least one float.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isTwoPointerTwoFloatOneIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .f32, .f32, .u32), such as Monte Carlo integration.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t sampleBytes = static_cast<size_t>(sampleCount) * sizeof(float);
        const size_t outputBytes = sizeof(float);
        const float zeroOutput = 0.0f;
        const CUdeviceptr samplesPtr = vm->allocateMemory(sampleBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(samplesPtr, samples, sampleBytes)) {
            lastError_ = "Failed to copy the sample values into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, &zeroOutput, outputBytes)) {
            lastError_ = "Failed to initialize the scalar output in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({samplesPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({packFloat32(lowerBound), kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({packFloat32(upperBound), kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(sampleCount), kernel->parameters[4].size, kernel->parameters[4].offset});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        float outputValue = 0.0f;
        if (!vm->copyMemoryDtoH(&outputValue, outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the Monte Carlo output failed.";
            return false;
        }

        lastResult_.assign(1, outputValue);
        return true;
    }

    bool runRmsNormalizationDemo(const std::string& kernelName,
                                 const float* input,
                                 int inputLen,
                                 float gamma,
                                 float beta) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (inputLen <= 0) {
            lastError_ = "Input buffer must contain at least one float.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isTwoPointerTwoFloatOneIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .f32, .f32, .u32), such as RMS normalization.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputLen) * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(static_cast<size_t>(inputLen), 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input vector into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output vector in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({packFloat32(gamma), kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({packFloat32(beta), kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(inputLen), kernel->parameters[4].size, kernel->parameters[4].offset});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(static_cast<size_t>(inputLen), 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the RMS normalization output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runSoftmaxDemo(const std::string& kernelName,
                        const float* input,
                        int inputLen) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (inputLen <= 0) {
            lastError_ = "Input array must contain at least one float.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isSoftmaxSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u32), such as softmax.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputLen) * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(static_cast<size_t>(inputLen), 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input array into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({
            static_cast<CUdeviceptr>(inputLen),
            kernel->parameters[2].size,
            kernel->parameters[2].offset,
        });

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(static_cast<size_t>(inputLen), 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the softmax output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runSoftmaxAttentionDemo(const std::string& kernelName,
                                 const float* inputQ,
                                 int inputQLen,
                                 const float* inputK,
                                 int inputKLen,
                                 const float* inputV,
                                 int inputVLen,
                                 int rowsM,
                                 int sharedN,
                                 int featureD) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputQ == nullptr || inputK == nullptr || inputV == nullptr) {
            lastError_ = "Q, K, and V buffers must not be null.";
            return false;
        }

        if (rowsM <= 0 || sharedN <= 0 || featureD <= 0) {
            lastError_ = "M, N, and d must all be positive integers.";
            return false;
        }

        const size_t expectedQLen =
            static_cast<size_t>(rowsM) * static_cast<size_t>(featureD);
        const size_t expectedKLen =
            static_cast<size_t>(sharedN) * static_cast<size_t>(featureD);
        const size_t expectedVLen = expectedKLen;
        const size_t outputElementCount = expectedQLen;

        if (static_cast<size_t>(inputQLen) != expectedQLen) {
            lastError_ = "Matrix Q does not match the provided M x d dimensions.";
            return false;
        }

        if (static_cast<size_t>(inputKLen) != expectedKLen) {
            lastError_ = "Matrix K does not match the provided N x d dimensions.";
            return false;
        }

        if (static_cast<size_t>(inputVLen) != expectedVLen) {
            lastError_ = "Matrix V does not match the provided N x d dimensions.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isSoftmaxAttentionSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u64, .u32, .u32, .u32), such as softmax attention.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputQBytes = expectedQLen * sizeof(float);
        const size_t inputKBytes = expectedKLen * sizeof(float);
        const size_t inputVBytes = expectedVLen * sizeof(float);
        const size_t outputBytes = outputElementCount * sizeof(float);
        const std::vector<float> zeroOutput(outputElementCount, 0.0f);

        const CUdeviceptr inputQPtr = vm->allocateMemory(inputQBytes);
        const CUdeviceptr inputKPtr = vm->allocateMemory(inputKBytes);
        const CUdeviceptr inputVPtr = vm->allocateMemory(inputVBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputQPtr, inputQ, inputQBytes)) {
            lastError_ = "Failed to copy Matrix Q into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputKPtr, inputK, inputKBytes)) {
            lastError_ = "Failed to copy Matrix K into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputVPtr, inputV, inputVBytes)) {
            lastError_ = "Failed to copy Matrix V into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output matrix in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputQPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({inputKPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({inputVPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({outputPtr, kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({
            static_cast<CUdeviceptr>(rowsM),
            kernel->parameters[4].size,
            kernel->parameters[4].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(sharedN),
            kernel->parameters[5].size,
            kernel->parameters[5].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(featureD),
            kernel->parameters[6].size,
            kernel->parameters[6].offset,
        });

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(outputElementCount, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the attention output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runLinearSelfAttentionDemo(const std::string& kernelName,
                                    const float* inputQ,
                                    int inputQLen,
                                    const float* inputK,
                                    int inputKLen,
                                    const float* inputV,
                                    int inputVLen,
                                    int rowsM,
                                    int featureD) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputQ == nullptr || inputK == nullptr || inputV == nullptr) {
            lastError_ = "Q, K, and V buffers must not be null.";
            return false;
        }

        if (rowsM <= 0 || featureD <= 0) {
            lastError_ = "M and d must both be positive integers.";
            return false;
        }

        const size_t expectedLen =
            static_cast<size_t>(rowsM) * static_cast<size_t>(featureD);
        if (static_cast<size_t>(inputQLen) != expectedLen ||
            static_cast<size_t>(inputKLen) != expectedLen ||
            static_cast<size_t>(inputVLen) != expectedLen) {
            lastError_ = "Q, K, and V must all match the provided M x d dimensions.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isFourPointerTwoIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u64, .u32, .u32), such as linear self-attention.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedLen * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(expectedLen, 0.0f);
        const CUdeviceptr inputQPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr inputKPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr inputVPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputQPtr, inputQ, inputBytes)) {
            lastError_ = "Failed to copy Matrix Q into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputKPtr, inputK, inputBytes)) {
            lastError_ = "Failed to copy Matrix K into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputVPtr, inputV, inputBytes)) {
            lastError_ = "Failed to copy Matrix V into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output matrix in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputQPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({inputKPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({inputVPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({outputPtr, kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(rowsM), kernel->parameters[4].size, kernel->parameters[4].offset});
        params.push_back({static_cast<CUdeviceptr>(featureD), kernel->parameters[5].size, kernel->parameters[5].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(expectedLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the linear attention output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runRotaryEmbeddingDemo(const std::string& kernelName,
                                const float* inputQ,
                                int inputQLen,
                                const float* inputCos,
                                int inputCosLen,
                                const float* inputSin,
                                int inputSinLen,
                                int rowsM,
                                int featureD) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputQ == nullptr || inputCos == nullptr || inputSin == nullptr) {
            lastError_ = "Q, cos, and sin buffers must not be null.";
            return false;
        }

        if (rowsM <= 0 || featureD <= 0 || (featureD % 2) != 0) {
            lastError_ = "M must be positive and D must be a positive even integer.";
            return false;
        }

        const size_t expectedLen =
            static_cast<size_t>(rowsM) * static_cast<size_t>(featureD);
        if (static_cast<size_t>(inputQLen) != expectedLen ||
            static_cast<size_t>(inputCosLen) != expectedLen ||
            static_cast<size_t>(inputSinLen) != expectedLen) {
            lastError_ = "Q, cos, and sin must all match the provided M x D dimensions.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isFourPointerTwoIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u64, .u32, .u32), such as rotary positional embedding.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedLen * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(expectedLen, 0.0f);
        const CUdeviceptr inputQPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr inputCosPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr inputSinPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputQPtr, inputQ, inputBytes)) {
            lastError_ = "Failed to copy Matrix Q into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputCosPtr, inputCos, inputBytes)) {
            lastError_ = "Failed to copy Matrix cos into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputSinPtr, inputSin, inputBytes)) {
            lastError_ = "Failed to copy Matrix sin into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output matrix in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputQPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({inputCosPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({inputSinPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({outputPtr, kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(rowsM), kernel->parameters[4].size, kernel->parameters[4].offset});
        params.push_back({static_cast<CUdeviceptr>(featureD), kernel->parameters[5].size, kernel->parameters[5].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(expectedLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the rotary embedding output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runMultiHeadAttentionDemo(const std::string& kernelName,
                                   const float* inputQ,
                                   int inputQLen,
                                   const float* inputK,
                                   int inputKLen,
                                   const float* inputV,
                                   int inputVLen,
                                   int sequenceLen,
                                   int modelDim,
                                   int headCount) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputQ == nullptr || inputK == nullptr || inputV == nullptr) {
            lastError_ = "Q, K, and V buffers must not be null.";
            return false;
        }

        if (sequenceLen <= 0 || modelDim <= 0 || headCount <= 0) {
            lastError_ = "N, d_model, and h must all be positive integers.";
            return false;
        }

        if (headCount > modelDim || (modelDim % headCount) != 0) {
            lastError_ = "d_model must be divisible by h, and h must not exceed d_model.";
            return false;
        }

        const size_t expectedLen =
            static_cast<size_t>(sequenceLen) * static_cast<size_t>(modelDim);

        if (static_cast<size_t>(inputQLen) != expectedLen) {
            lastError_ = "Matrix Q does not match the provided N x d_model dimensions.";
            return false;
        }

        if (static_cast<size_t>(inputKLen) != expectedLen) {
            lastError_ = "Matrix K does not match the provided N x d_model dimensions.";
            return false;
        }

        if (static_cast<size_t>(inputVLen) != expectedLen) {
            lastError_ = "Matrix V does not match the provided N x d_model dimensions.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isMultiHeadAttentionSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u64, .u32, .u32, .u32), such as multi-head attention.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedLen * sizeof(float);
        const size_t outputBytes = expectedLen * sizeof(float);
        const std::vector<float> zeroOutput(expectedLen, 0.0f);

        const CUdeviceptr inputQPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr inputKPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr inputVPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputQPtr, inputQ, inputBytes)) {
            lastError_ = "Failed to copy Matrix Q into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputKPtr, inputK, inputBytes)) {
            lastError_ = "Failed to copy Matrix K into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputVPtr, inputV, inputBytes)) {
            lastError_ = "Failed to copy Matrix V into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output matrix in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputQPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({inputKPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({inputVPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({outputPtr, kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({
            static_cast<CUdeviceptr>(sequenceLen),
            kernel->parameters[4].size,
            kernel->parameters[4].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(modelDim),
            kernel->parameters[5].size,
            kernel->parameters[5].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(headCount),
            kernel->parameters[6].size,
            kernel->parameters[6].offset,
        });

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(expectedLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the multi-head attention output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runCausalSelfAttentionDemo(const std::string& kernelName,
                                    const float* inputQ,
                                    int inputQLen,
                                    const float* inputK,
                                    int inputKLen,
                                    const float* inputV,
                                    int inputVLen,
                                    int rowsM,
                                    int featureD) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputQ == nullptr || inputK == nullptr || inputV == nullptr) {
            lastError_ = "Q, K, and V buffers must not be null.";
            return false;
        }

        if (rowsM <= 0 || featureD <= 0) {
            lastError_ = "M and d must both be positive integers.";
            return false;
        }

        const size_t expectedLen =
            static_cast<size_t>(rowsM) * static_cast<size_t>(featureD);
        if (static_cast<size_t>(inputQLen) != expectedLen ||
            static_cast<size_t>(inputKLen) != expectedLen ||
            static_cast<size_t>(inputVLen) != expectedLen) {
            lastError_ = "Q, K, and V must all match the provided M x d dimensions.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isFourPointerTwoIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u64, .u32, .u32), such as causal self-attention.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedLen * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(expectedLen, 0.0f);
        const CUdeviceptr inputQPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr inputKPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr inputVPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputQPtr, inputQ, inputBytes)) {
            lastError_ = "Failed to copy Matrix Q into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputKPtr, inputK, inputBytes)) {
            lastError_ = "Failed to copy Matrix K into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputVPtr, inputV, inputBytes)) {
            lastError_ = "Failed to copy Matrix V into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output matrix in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputQPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({inputKPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({inputVPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({outputPtr, kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(rowsM), kernel->parameters[4].size, kernel->parameters[4].offset});
        params.push_back({static_cast<CUdeviceptr>(featureD), kernel->parameters[5].size, kernel->parameters[5].offset});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(expectedLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the causal attention output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runSlidingWindowAttentionDemo(const std::string& kernelName,
                                       const float* inputQ,
                                       int inputQLen,
                                       const float* inputK,
                                       int inputKLen,
                                       const float* inputV,
                                       int inputVLen,
                                       int rowsM,
                                       int featureD,
                                       int windowSize) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputQ == nullptr || inputK == nullptr || inputV == nullptr) {
            lastError_ = "Q, K, and V buffers must not be null.";
            return false;
        }

        if (rowsM <= 0 || featureD <= 0 || windowSize < 0) {
            lastError_ = "M and d must be positive, and window_size must be non-negative.";
            return false;
        }

        const size_t expectedLen =
            static_cast<size_t>(rowsM) * static_cast<size_t>(featureD);
        if (static_cast<size_t>(inputQLen) != expectedLen ||
            static_cast<size_t>(inputKLen) != expectedLen ||
            static_cast<size_t>(inputVLen) != expectedLen) {
            lastError_ = "Q, K, and V must all match the provided M x d dimensions.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isFourPointerThreeIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u64, .u32, .u32, .u32), such as sliding-window attention.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedLen * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(expectedLen, 0.0f);
        const CUdeviceptr inputQPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr inputKPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr inputVPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputQPtr, inputQ, inputBytes)) {
            lastError_ = "Failed to copy Matrix Q into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputKPtr, inputK, inputBytes)) {
            lastError_ = "Failed to copy Matrix K into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputVPtr, inputV, inputBytes)) {
            lastError_ = "Failed to copy Matrix V into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output matrix in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputQPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({inputKPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({inputVPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({outputPtr, kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(rowsM), kernel->parameters[4].size, kernel->parameters[4].offset});
        params.push_back({static_cast<CUdeviceptr>(featureD), kernel->parameters[5].size, kernel->parameters[5].offset});
        params.push_back({static_cast<CUdeviceptr>(windowSize), kernel->parameters[6].size, kernel->parameters[6].offset});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(expectedLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the sliding-window attention output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runAlibiAttentionDemo(const std::string& kernelName,
                               const float* inputQ,
                               int inputQLen,
                               const float* inputK,
                               int inputKLen,
                               const float* inputV,
                               int inputVLen,
                               int rowsM,
                               int sharedN,
                               int featureD,
                               float alpha) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputQ == nullptr || inputK == nullptr || inputV == nullptr) {
            lastError_ = "Q, K, and V buffers must not be null.";
            return false;
        }

        if (rowsM <= 0 || sharedN <= 0 || featureD <= 0) {
            lastError_ = "M, N, and d must all be positive integers.";
            return false;
        }

        const size_t expectedQLen =
            static_cast<size_t>(rowsM) * static_cast<size_t>(featureD);
        const size_t expectedKVLen =
            static_cast<size_t>(sharedN) * static_cast<size_t>(featureD);
        const size_t expectedOutputLen =
            static_cast<size_t>(rowsM) * static_cast<size_t>(featureD);
        if (static_cast<size_t>(inputQLen) != expectedQLen ||
            static_cast<size_t>(inputKLen) != expectedKVLen ||
            static_cast<size_t>(inputVLen) != expectedKVLen) {
            lastError_ = "Q, K, and V must match the provided M x d and N x d dimensions.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isFourPointerThreeIntOneFloatSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u64, .u32, .u32, .u32, .f32), such as ALiBi attention.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputQBytes = expectedQLen * sizeof(float);
        const size_t inputKBytes = expectedKVLen * sizeof(float);
        const size_t inputVBytes = expectedKVLen * sizeof(float);
        const size_t outputBytes = expectedOutputLen * sizeof(float);
        const std::vector<float> zeroOutput(expectedOutputLen, 0.0f);
        const CUdeviceptr inputQPtr = vm->allocateMemory(inputQBytes);
        const CUdeviceptr inputKPtr = vm->allocateMemory(inputKBytes);
        const CUdeviceptr inputVPtr = vm->allocateMemory(inputVBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputQPtr, inputQ, inputQBytes)) {
            lastError_ = "Failed to copy Matrix Q into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputKPtr, inputK, inputKBytes)) {
            lastError_ = "Failed to copy Matrix K into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputVPtr, inputV, inputVBytes)) {
            lastError_ = "Failed to copy Matrix V into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output matrix in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputQPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({inputKPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({inputVPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({outputPtr, kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(rowsM), kernel->parameters[4].size, kernel->parameters[4].offset});
        params.push_back({static_cast<CUdeviceptr>(sharedN), kernel->parameters[5].size, kernel->parameters[5].offset});
        params.push_back({static_cast<CUdeviceptr>(featureD), kernel->parameters[6].size, kernel->parameters[6].offset});
        params.push_back({packFloat32(alpha), kernel->parameters[7].size, kernel->parameters[7].offset});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(expectedOutputLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the ALiBi attention output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runWeightDequantizationDemo(const std::string& kernelName,
                                     const float* inputA,
                                     int inputALen,
                                     const float* inputB,
                                     int inputBLen,
                                     int rowsM,
                                     int colsN,
                                     int tileSize) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputA == nullptr || inputB == nullptr) {
            lastError_ = "Input buffers must not be null.";
            return false;
        }

        if (rowsM <= 0 || colsN <= 0 || tileSize <= 0) {
            lastError_ = "M, N, and TILE_SIZE must all be positive integers.";
            return false;
        }

        const size_t expectedALen =
            static_cast<size_t>(rowsM) * static_cast<size_t>(colsN);
        const size_t scaleRows =
            (static_cast<size_t>(rowsM) + static_cast<size_t>(tileSize) - 1u) / static_cast<size_t>(tileSize);
        const size_t scaleCols =
            (static_cast<size_t>(colsN) + static_cast<size_t>(tileSize) - 1u) / static_cast<size_t>(tileSize);
        const size_t expectedBLen = scaleRows * scaleCols;
        const size_t outputLen = expectedALen;

        if (static_cast<size_t>(inputALen) != expectedALen) {
            lastError_ = "Matrix X does not match the provided M x N dimensions.";
            return false;
        }

        if (static_cast<size_t>(inputBLen) != expectedBLen) {
            lastError_ = "Scale matrix S does not match ceil(M / TILE_SIZE) x ceil(N / TILE_SIZE).";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isMatrixMultiplySignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32, .u32, .u32), such as weight dequantization.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputABytes = expectedALen * sizeof(float);
        const size_t inputBBytes = expectedBLen * sizeof(float);
        const size_t outputBytes = outputLen * sizeof(float);
        const std::vector<float> zeroOutput(outputLen, 0.0f);
        const CUdeviceptr inputAPtr = vm->allocateMemory(inputABytes);
        const CUdeviceptr inputBPtr = vm->allocateMemory(inputBBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputAPtr, inputA, inputABytes)) {
            lastError_ = "Failed to copy Matrix X into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputBPtr, inputB, inputBBytes)) {
            lastError_ = "Failed to copy Matrix S into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output matrix in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputAPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({inputBPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({outputPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({static_cast<CUdeviceptr>(rowsM), kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(colsN), kernel->parameters[4].size, kernel->parameters[4].offset});
        params.push_back({static_cast<CUdeviceptr>(tileSize), kernel->parameters[5].size, kernel->parameters[5].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(outputLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the dequantized output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runValueClippingDemo(const std::string& kernelName,
                              const float* input,
                              int inputLen,
                              float lowerBound,
                              float upperBound) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (inputLen <= 0) {
            lastError_ = "Input buffer must contain at least one float.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isTwoPointerTwoFloatOneIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .f32, .f32, .u32), such as value clipping.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputLen) * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(static_cast<size_t>(inputLen), 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input vector into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output vector in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({packFloat32(lowerBound), kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({packFloat32(upperBound), kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(inputLen), kernel->parameters[4].size, kernel->parameters[4].offset});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(static_cast<size_t>(inputLen), 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the clipped output failed.";
            lastResult_.clear();
            return false;
        }

        return true;
    }

    bool runColorInversionDemo(const std::string& kernelName,
                               const std::uint8_t* image,
                               int imageLen,
                               int width,
                               int height) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (image == nullptr) {
            lastError_ = "Image buffer must not be null.";
            return false;
        }

        if (width <= 0 || height <= 0) {
            lastError_ = "Width and height must both be positive integers.";
            return false;
        }

        const size_t expectedLen =
            static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
        if (static_cast<size_t>(imageLen) != expectedLen) {
            lastError_ = "Image length must equal width x height x 4 RGBA bytes.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isColorInversionSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u32, .u32), such as color inversion.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t imageBytes = expectedLen * sizeof(std::uint8_t);
        const CUdeviceptr imagePtr = vm->allocateMemory(imageBytes);
        if (!vm->copyMemoryHtoD(imagePtr, image, imageBytes)) {
            lastError_ = "Failed to copy the image buffer into VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({imagePtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({
            static_cast<CUdeviceptr>(width),
            kernel->parameters[1].size,
            kernel->parameters[1].offset,
        });
        params.push_back({
            static_cast<CUdeviceptr>(height),
            kernel->parameters[2].size,
            kernel->parameters[2].offset,
        });

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        std::vector<std::uint8_t> outputImage(expectedLen, 0);
        if (!vm->copyMemoryDtoH(outputImage.data(), imagePtr, imageBytes)) {
            lastError_ = "Kernel executed, but reading the image output failed.";
            return false;
        }

        lastResult_.assign(expectedLen, 0.0f);
        for (size_t i = 0; i < expectedLen; ++i) {
            lastResult_[i] = static_cast<float>(outputImage[i]);
        }

        return true;
    }

    bool runElementwiseDemo(const std::string& kernelName,
                           const float* input,
                           int inputLen) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (inputLen <= 0) {
            lastError_ = "Input array must contain at least one float.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputLen) * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(static_cast<size_t>(inputLen), 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input array into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, 8, 0});
        params.push_back({outputPtr, 8, 8});
        params.push_back({static_cast<CUdeviceptr>(inputLen), 4, 16});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(static_cast<size_t>(inputLen), 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the output failed.";
            return false;
        }

        return true;
    }

    bool runElementwiseIntDemo(const std::string& kernelName,
                               const std::int32_t* input,
                               int inputLen) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (inputLen <= 0) {
            lastError_ = "Input array must contain at least one int.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputLen) * sizeof(std::int32_t);
        const size_t outputBytes = inputBytes;
        const std::vector<std::int32_t> zeroOutput(static_cast<size_t>(inputLen), 0);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input array into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, 8, 0});
        params.push_back({outputPtr, 8, 8});
        params.push_back({static_cast<CUdeviceptr>(inputLen), 4, 16});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        std::vector<std::int32_t> output(static_cast<size_t>(inputLen), 0);
        if (!vm->copyMemoryDtoH(output.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the output failed.";
            return false;
        }

        lastResult_.assign(static_cast<size_t>(inputLen), 0.0f);
        for (int i = 0; i < inputLen; ++i) {
            lastResult_[static_cast<size_t>(i)] = static_cast<float>(output[static_cast<size_t>(i)]);
        }

        return true;
    }

    bool runIntArrayTransformDemo(const std::string& kernelName,
                                  const std::int32_t* input,
                                  int inputLen) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (inputLen <= 0) {
            lastError_ = "Input array must contain at least one integer.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isSquareMatrixTransformSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u32), such as radix sort.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputLen) * sizeof(std::int32_t);
        const size_t outputBytes = inputBytes;
        const std::vector<std::int32_t> zeroOutput(static_cast<size_t>(inputLen), 0);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input array into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({static_cast<CUdeviceptr>(inputLen), kernel->parameters[2].size, kernel->parameters[2].offset});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        std::vector<std::int32_t> output(static_cast<size_t>(inputLen), 0);
        if (!vm->copyMemoryDtoH(output.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the integer output failed.";
            return false;
        }

        lastResult_.assign(static_cast<size_t>(inputLen), 0.0f);
        for (int i = 0; i < inputLen; ++i) {
            lastResult_[static_cast<size_t>(i)] = static_cast<float>(output[static_cast<size_t>(i)]);
        }

        return true;
    }

    bool runRainbowTableDemo(const std::string& kernelName,
                             const std::int32_t* input,
                             int inputLen,
                             int rounds) {
        lastError_.clear();
        lastResult_.clear();
        lastUInt32Result_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (inputLen <= 0) {
            lastError_ = "Input array must contain at least one integer.";
            return false;
        }

        if (rounds <= 0) {
            lastError_ = "R must be a positive integer.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isTwoPointerTwoIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u32, .u32), such as rainbow table hashing.";
            return false;
        }

        lastUInt32Result_.assign(static_cast<size_t>(inputLen), 0);

        for (int i = 0; i < inputLen; ++i) {
            auto vm = createVm();
            if (!vm) {
                return false;
            }

            if (!vm->loadProgram(loadedPath_)) {
                lastError_ = "Failed to reload the uploaded PTX program.";
                lastUInt32Result_.clear();
                return false;
            }

            const size_t inputBytes = sizeof(std::int32_t);
            const size_t outputBytes = sizeof(std::uint32_t);
            const std::uint32_t zeroOutput = 0;
            const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
            const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

            if (!vm->copyMemoryHtoD(inputPtr, &input[i], inputBytes)) {
                lastError_ = "Failed to copy the input value into VM memory.";
                lastUInt32Result_.clear();
                return false;
            }

            if (!vm->copyMemoryHtoD(outputPtr, &zeroOutput, outputBytes)) {
                lastError_ = "Failed to initialize the output hash buffer in VM memory.";
                lastUInt32Result_.clear();
                return false;
            }

            std::vector<KernelParameter> params;
            params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
            params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
            params.push_back({static_cast<CUdeviceptr>(1), kernel->parameters[2].size, kernel->parameters[2].offset});
            params.push_back({static_cast<CUdeviceptr>(rounds), kernel->parameters[3].size, kernel->parameters[3].offset});

            vm->setKernelParameters(params);
            vm->getExecutor().setGridDimensions(1, 1, 1, 32, 1, 1);

            if (!vm->run()) {
                lastError_ = "Kernel execution failed inside the PTX VM.";
                lastUInt32Result_.clear();
                return false;
            }

            std::uint32_t outputValue = 0;
            if (!vm->copyMemoryDtoH(&outputValue, outputPtr, outputBytes)) {
                lastError_ = "Kernel executed, but reading the output hash failed.";
                lastUInt32Result_.clear();
                return false;
            }

            lastUInt32Result_[static_cast<size_t>(i)] = outputValue;
        }

        return true;
    }

    bool runMatrixCopyDemo(const std::string& kernelName,
                          const float* input,
                          int inputLen) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (inputLen <= 0) {
            lastError_ = "Input array must contain at least one float.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputLen) * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(static_cast<size_t>(inputLen), 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input array into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, 8, 0});
        params.push_back({outputPtr, 8, 8});
        params.push_back({static_cast<CUdeviceptr>(inputLen), 4, 16});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(static_cast<size_t>(inputLen), 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the output failed.";
            return false;
        }

        return true;
    }

    bool runMatrixPowerDemo(const std::string& kernelName,
                            const float* input,
                            int inputLen,
                            int matrixSize,
                            int power) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (matrixSize <= 0 || power <= 0) {
            lastError_ = "Matrix size N and power P must both be positive integers.";
            return false;
        }

        const size_t expectedLen = static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize);
        if (static_cast<size_t>(inputLen) != expectedLen) {
            lastError_ = "Input length does not match N x N.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isMatrixTransposeSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u32, .u32), such as matrix power.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedLen * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(expectedLen, 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input matrix into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output matrix in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({static_cast<CUdeviceptr>(matrixSize), kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({static_cast<CUdeviceptr>(power), kernel->parameters[3].size, kernel->parameters[3].offset});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(expectedLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the matrix power output failed.";
            return false;
        }

        return true;
    }

    bool runSquareMatrixTransformDemo(const std::string& kernelName,
                                      const float* input,
                                      int inputLen,
                                      int matrixSize) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (matrixSize <= 0) {
            lastError_ = "Matrix size N must be a positive integer.";
            return false;
        }

        const size_t expectedLen = static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize);
        if (static_cast<size_t>(inputLen) != expectedLen) {
            lastError_ = "Input length does not match N x N.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isSquareMatrixTransformSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u32), such as all-pairs shortest paths.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedLen * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(expectedLen, 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input matrix into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output matrix in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({static_cast<CUdeviceptr>(matrixSize), kernel->parameters[2].size, kernel->parameters[2].offset});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(expectedLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the matrix output failed.";
            return false;
        }

        return true;
    }

    bool runDotProductDemo(const std::string& kernelName,
                           const float* inputA,
                           int inputALen,
                           const float* inputB,
                           int inputBLen) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputA == nullptr || inputB == nullptr) {
            lastError_ = "Input buffers must not be null.";
            return false;
        }

        if (inputALen <= 0 || inputBLen <= 0) {
            lastError_ = "Both input arrays must contain at least one float.";
            return false;
        }

        if (inputALen != inputBLen) {
            lastError_ = "Input A and Input B must have the same length.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputALen) * sizeof(float);
        const size_t outputBytes = sizeof(float);
        const float zeroOutput = 0.0f;
        const CUdeviceptr inputAPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr inputBPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputAPtr, inputA, inputBytes)) {
            lastError_ = "Failed to copy Input A into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputBPtr, inputB, inputBytes)) {
            lastError_ = "Failed to copy Input B into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, &zeroOutput, outputBytes)) {
            lastError_ = "Failed to initialize the output buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputAPtr, 8, 0});
        params.push_back({inputBPtr, 8, 8});
        params.push_back({outputPtr, 8, 16});
        params.push_back({static_cast<CUdeviceptr>(inputALen), 4, 24});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        float outputValue = 0.0f;
        if (!vm->copyMemoryDtoH(&outputValue, outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the dot product output failed.";
            return false;
        }

        lastResult_.assign(1, outputValue);
        return true;
    }

    bool runPrefixSumDemo(const std::string& kernelName,
                         const float* input,
                         int inputLen) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (inputLen <= 0) {
            lastError_ = "Input array must contain at least one float.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputLen) * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(static_cast<size_t>(inputLen), 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input array into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, 8, 0});
        params.push_back({outputPtr, 8, 8});
        params.push_back({static_cast<CUdeviceptr>(inputLen), 4, 16});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(static_cast<size_t>(inputLen), 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the output failed.";
            return false;
        }

        return true;
    }

    bool runReverseArrayDemo(const std::string& kernelName,
                             float* input,
                             int inputLen) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (inputLen <= 0) {
            lastError_ = "Input array must contain at least one float.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputLen) * sizeof(float);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input array into VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, 8, 0});
        params.push_back({static_cast<CUdeviceptr>(inputLen), 4, 8});

        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        std::vector<float> output(static_cast<size_t>(inputLen), 0.0f);
        if (!vm->copyMemoryDtoH(output.data(), inputPtr, inputBytes)) {
            lastError_ = "Kernel executed, but reading the output failed.";
            return false;
        }

        lastResult_ = output;
        return true;
    }

    bool runSortingDemo(const std::string& kernelName,
                        float* input,
                        int inputLen) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (inputLen <= 0) {
            lastError_ = "Input array must contain at least one float.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isInPlaceArraySignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u32), such as in-place sorting.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputLen) * sizeof(float);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input array into VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({static_cast<CUdeviceptr>(inputLen), kernel->parameters[1].size, kernel->parameters[1].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(static_cast<size_t>(inputLen), 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), inputPtr, inputBytes)) {
            lastError_ = "Kernel executed, but reading the sorted output failed.";
            return false;
        }

        return true;
    }

    bool runTopKDemo(const std::string& kernelName,
                     const float* input,
                     int inputLen,
                     int k) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (inputLen <= 0 || k <= 0) {
            lastError_ = "Input array and k must both be positive.";
            return false;
        }

        if (k > inputLen) {
            lastError_ = "k must not exceed the input length.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isMatrixTransposeSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u32, .u32), such as top-k selection.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputLen) * sizeof(float);
        const size_t outputBytes = static_cast<size_t>(k) * sizeof(float);
        const std::vector<float> zeroOutput(static_cast<size_t>(k), 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input array into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({static_cast<CUdeviceptr>(inputLen), kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({static_cast<CUdeviceptr>(k), kernel->parameters[3].size, kernel->parameters[3].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(static_cast<size_t>(k), 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the top-k output failed.";
            return false;
        }

        return true;
    }

    bool runTopPSamplingDemo(const std::string& kernelName,
                             const float* logits,
                             int vocabSize,
                             const float* pValue,
                             const std::int32_t* seedValue) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (logits == nullptr || pValue == nullptr || seedValue == nullptr) {
            lastError_ = "Input buffers must not be null.";
            return false;
        }

        if (vocabSize <= 0) {
            lastError_ = "vocab_size must be a positive integer.";
            return false;
        }

        if (!(pValue[0] > 0.0f && pValue[0] <= 1.0f)) {
            lastError_ = "p must satisfy 0 < p <= 1.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isFourPointerOneIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u64, .u32), such as top-p sampling.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t logitsBytes = static_cast<size_t>(vocabSize) * sizeof(float);
        const size_t pBytes = sizeof(float);
        const size_t seedBytes = sizeof(std::int32_t);
        const size_t outputBytes = sizeof(std::int32_t);
        const std::int32_t zeroOutput = 0;
        const CUdeviceptr logitsPtr = vm->allocateMemory(logitsBytes);
        const CUdeviceptr pPtr = vm->allocateMemory(pBytes);
        const CUdeviceptr seedPtr = vm->allocateMemory(seedBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(logitsPtr, logits, logitsBytes) ||
            !vm->copyMemoryHtoD(pPtr, pValue, pBytes) ||
            !vm->copyMemoryHtoD(seedPtr, seedValue, seedBytes)) {
            lastError_ = "Failed to copy the sampling inputs into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, &zeroOutput, outputBytes)) {
            lastError_ = "Failed to initialize the sampled_token buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({logitsPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({pPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({seedPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({outputPtr, kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(vocabSize), kernel->parameters[4].size, kernel->parameters[4].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        std::int32_t sampledToken = 0;
        if (!vm->copyMemoryDtoH(&sampledToken, outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the sampled_token output failed.";
            return false;
        }

        lastResult_.assign(1, static_cast<float>(sampledToken));
        return true;
    }

    bool runMoeTopKGatingDemo(const std::string& kernelName,
                              const float* logits,
                              int logitsLen,
                              int rowsM,
                              int expertCount,
                              int topK) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (logits == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (rowsM <= 0 || expertCount <= 0 || topK <= 0) {
            lastError_ = "M, E, and k must all be positive integers.";
            return false;
        }

        if (topK > expertCount) {
            lastError_ = "k must not exceed E.";
            return false;
        }

        const size_t expectedLogitsLen = static_cast<size_t>(rowsM) * static_cast<size_t>(expertCount);
        if (static_cast<size_t>(logitsLen) != expectedLogitsLen) {
            lastError_ = "Logits length must equal M x E.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isThreePointerThreeIntSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32, .u32, .u32), such as MoE top-k gating.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t logitsBytes = expectedLogitsLen * sizeof(float);
        const size_t weightBytes = static_cast<size_t>(rowsM) * static_cast<size_t>(topK) * sizeof(float);
        const size_t indexBytes = static_cast<size_t>(rowsM) * static_cast<size_t>(topK) * sizeof(std::int32_t);
        const std::vector<float> zeroWeights(static_cast<size_t>(rowsM) * static_cast<size_t>(topK), 0.0f);
        const std::vector<std::int32_t> minusOneIndices(static_cast<size_t>(rowsM) * static_cast<size_t>(topK), -1);
        const CUdeviceptr logitsPtr = vm->allocateMemory(logitsBytes);
        const CUdeviceptr weightsPtr = vm->allocateMemory(weightBytes);
        const CUdeviceptr indicesPtr = vm->allocateMemory(indexBytes);

        if (!vm->copyMemoryHtoD(logitsPtr, logits, logitsBytes)) {
            lastError_ = "Failed to copy the logits matrix into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(weightsPtr, zeroWeights.data(), weightBytes) ||
            !vm->copyMemoryHtoD(indicesPtr, minusOneIndices.data(), indexBytes)) {
            lastError_ = "Failed to initialize the MoE output buffers in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({logitsPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({weightsPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({indicesPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({static_cast<CUdeviceptr>(rowsM), kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(expertCount), kernel->parameters[4].size, kernel->parameters[4].offset});
        params.push_back({static_cast<CUdeviceptr>(topK), kernel->parameters[5].size, kernel->parameters[5].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        std::vector<float> weights(static_cast<size_t>(rowsM) * static_cast<size_t>(topK), 0.0f);
        std::vector<std::int32_t> indices(static_cast<size_t>(rowsM) * static_cast<size_t>(topK), -1);
        if (!vm->copyMemoryDtoH(weights.data(), weightsPtr, weightBytes) ||
            !vm->copyMemoryDtoH(indices.data(), indicesPtr, indexBytes)) {
            lastError_ = "Kernel executed, but reading the MoE output buffers failed.";
            return false;
        }

        lastResult_.clear();
        lastResult_.reserve(weights.size() + indices.size());
        lastResult_.insert(lastResult_.end(), weights.begin(), weights.end());
        for (std::int32_t value : indices) {
            lastResult_.push_back(static_cast<float>(value));
        }
        return true;
    }

    bool runIntScalarReduceDemo(const std::string& kernelName,
                                const std::int32_t* input,
                                int inputLen,
                                const std::vector<int>& scalars) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (inputLen <= 0) {
            lastError_ = "Input array must contain at least one integer.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isIntScalarReduceSignature(*kernel, static_cast<int>(scalars.size()))) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, scalar...), returning one integer output.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputLen) * sizeof(std::int32_t);
        const size_t outputBytes = sizeof(std::int32_t);
        const std::int32_t zeroOutput = 0;
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input array into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, &zeroOutput, outputBytes)) {
            lastError_ = "Failed to initialize the output buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        for (size_t i = 0; i < scalars.size(); ++i) {
            params.push_back({
                static_cast<CUdeviceptr>(scalars[i]),
                kernel->parameters[i + 2].size,
                kernel->parameters[i + 2].offset,
            });
        }
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        std::int32_t outputValue = 0;
        if (!vm->copyMemoryDtoH(&outputValue, outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the integer output failed.";
            return false;
        }

        lastResult_.assign(1, static_cast<float>(outputValue));
        return true;
    }

    bool runInterleaveDemo(const std::string& kernelName,
                           const float* inputA,
                           int inputALen,
                           const float* inputB,
                           int inputBLen) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputA == nullptr || inputB == nullptr) {
            lastError_ = "Input buffers must not be null.";
            return false;
        }

        if (inputALen <= 0 || inputBLen <= 0 || inputALen != inputBLen) {
            lastError_ = "Input A and Input B must both be non-empty and have the same length.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isVectorAddSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32), such as array interleave.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = static_cast<size_t>(inputALen) * sizeof(float);
        const size_t outputLen = static_cast<size_t>(inputALen) * 2;
        const size_t outputBytes = outputLen * sizeof(float);
        const std::vector<float> zeroOutput(outputLen, 0.0f);
        const CUdeviceptr inputAPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr inputBPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputAPtr, inputA, inputBytes)) {
            lastError_ = "Failed to copy Input A into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputBPtr, inputB, inputBytes)) {
            lastError_ = "Failed to copy Input B into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputAPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({inputBPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({outputPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({static_cast<CUdeviceptr>(inputALen), kernel->parameters[3].size, kernel->parameters[3].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(outputLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the interleaved output failed.";
            return false;
        }

        return true;
    }

    bool runGrayscaleDemo(const std::string& kernelName,
                          const float* input,
                          int inputLen,
                          int width,
                          int height) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (width <= 0 || height <= 0) {
            lastError_ = "Width and height must both be positive integers.";
            return false;
        }

        const size_t expectedInputLen = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
        if (static_cast<size_t>(inputLen) != expectedInputLen) {
            lastError_ = "Input length must equal width x height x 3.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isMatrixTransposeSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u32, .u32), such as RGB to grayscale.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedInputLen * sizeof(float);
        const size_t outputLen = static_cast<size_t>(width) * static_cast<size_t>(height);
        const size_t outputBytes = outputLen * sizeof(float);
        const std::vector<float> zeroOutput(outputLen, 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the image into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the grayscale output buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({static_cast<CUdeviceptr>(width), kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({static_cast<CUdeviceptr>(height), kernel->parameters[3].size, kernel->parameters[3].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(outputLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the grayscale output failed.";
            return false;
        }

        return true;
    }

    bool runJacobi2DDemo(const std::string& kernelName,
                         const float* input,
                         int inputLen,
                         int rows,
                         int cols) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (input == nullptr) {
            lastError_ = "Input buffer must not be null.";
            return false;
        }

        if (rows <= 0 || cols <= 0) {
            lastError_ = "Rows and cols must both be positive integers.";
            return false;
        }

        const size_t expectedInputLen = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        if (static_cast<size_t>(inputLen) != expectedInputLen) {
            lastError_ = "Input length does not match rows x cols.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isMatrixTransposeSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u32, .u32), such as 2D Jacobi stencil.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t inputBytes = expectedInputLen * sizeof(float);
        const size_t outputBytes = inputBytes;
        const std::vector<float> zeroOutput(expectedInputLen, 0.0f);
        const CUdeviceptr inputPtr = vm->allocateMemory(inputBytes);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputPtr, input, inputBytes)) {
            lastError_ = "Failed to copy the input grid into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output grid in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({outputPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({static_cast<CUdeviceptr>(rows), kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({static_cast<CUdeviceptr>(cols), kernel->parameters[3].size, kernel->parameters[3].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(expectedInputLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the Jacobi output failed.";
            return false;
        }

        return true;
    }

    bool runMergeSortedDemo(const std::string& kernelName,
                            const float* inputA,
                            int inputALen,
                            const float* inputB,
                            int inputBLen) {
        lastError_.clear();
        lastResult_.clear();

        if (loadedPath_.empty()) {
            lastError_ = "Load a PTX file before launching the kernel.";
            return false;
        }

        if (inputA == nullptr || inputB == nullptr) {
            lastError_ = "Input buffers must not be null.";
            return false;
        }

        if (inputALen <= 0 || inputBLen <= 0) {
            lastError_ = "Both input arrays must contain at least one float.";
            return false;
        }

        auto validationVm = createVm();
        if (!validationVm) {
            return false;
        }

        if (!validationVm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const PTXProgram& program = validationVm->getExecutor().getProgram();
        const PTXFunction* kernel = resolveKernel(program, kernelName);
        if (kernel == nullptr) {
            lastError_ = "Could not find the requested kernel entry in the loaded PTX program.";
            return false;
        }

        if (!isConvolution1DSignature(*kernel)) {
            lastError_ =
                "This browser demo currently supports kernels with signature "
                "(.u64, .u64, .u64, .u32, .u32), such as merge sorted arrays.";
            return false;
        }

        auto vm = createVm();
        if (!vm) {
            return false;
        }

        if (!vm->loadProgram(loadedPath_)) {
            lastError_ = "Failed to reload the uploaded PTX program.";
            return false;
        }

        const size_t bytesA = static_cast<size_t>(inputALen) * sizeof(float);
        const size_t bytesB = static_cast<size_t>(inputBLen) * sizeof(float);
        const size_t outputLen = static_cast<size_t>(inputALen + inputBLen);
        const size_t outputBytes = outputLen * sizeof(float);
        const std::vector<float> zeroOutput(outputLen, 0.0f);
        const CUdeviceptr inputAPtr = vm->allocateMemory(bytesA);
        const CUdeviceptr inputBPtr = vm->allocateMemory(bytesB);
        const CUdeviceptr outputPtr = vm->allocateMemory(outputBytes);

        if (!vm->copyMemoryHtoD(inputAPtr, inputA, bytesA)) {
            lastError_ = "Failed to copy Input A into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(inputBPtr, inputB, bytesB)) {
            lastError_ = "Failed to copy Input B into VM memory.";
            return false;
        }

        if (!vm->copyMemoryHtoD(outputPtr, zeroOutput.data(), outputBytes)) {
            lastError_ = "Failed to initialize the output buffer in VM memory.";
            return false;
        }

        std::vector<KernelParameter> params;
        params.push_back({inputAPtr, kernel->parameters[0].size, kernel->parameters[0].offset});
        params.push_back({inputBPtr, kernel->parameters[1].size, kernel->parameters[1].offset});
        params.push_back({outputPtr, kernel->parameters[2].size, kernel->parameters[2].offset});
        params.push_back({static_cast<CUdeviceptr>(inputALen), kernel->parameters[3].size, kernel->parameters[3].offset});
        params.push_back({static_cast<CUdeviceptr>(inputBLen), kernel->parameters[4].size, kernel->parameters[4].offset});
        vm->setKernelParameters(params);

        PTXExecutor& executor = vm->getExecutor();
        executor.setGridDimensions(1, 1, 1, 1, 1, 1);

        ThreadExecutionContext context;
        context.gridDimX = 1;
        context.gridDimY = 1;
        context.gridDimZ = 1;
        context.blockDimX = 1;
        context.blockDimY = 1;
        context.blockDimZ = 1;
        context.blockIdxX = 0;
        context.blockIdxY = 0;
        context.blockIdxZ = 0;
        context.threadIdxX = 0;
        context.threadIdxY = 0;
        context.threadIdxZ = 0;
        context.warpSize = 32;
        context.laneId = 0;
        executor.setSingleThreadExecutionContext(context);

        if (!vm->run()) {
            lastError_ = "Kernel execution failed inside the PTX VM.";
            return false;
        }

        lastResult_.assign(outputLen, 0.0f);
        if (!vm->copyMemoryDtoH(lastResult_.data(), outputPtr, outputBytes)) {
            lastError_ = "Kernel executed, but reading the merged output failed.";
            return false;
        }

        return true;
    }

    int getResultCount() const {
        if (!lastResult_.empty()) {
            return static_cast<int>(lastResult_.size());
        }
        return static_cast<int>(lastUInt32Result_.size());
    }

    double getResultValue(int index) const {
        if (!lastResult_.empty()) {
            if (index < 0 || static_cast<size_t>(index) >= lastResult_.size()) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return static_cast<double>(lastResult_[static_cast<size_t>(index)]);
        }

        if (index < 0 || static_cast<size_t>(index) >= lastUInt32Result_.size()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return static_cast<double>(lastUInt32Result_[static_cast<size_t>(index)]);
    }

    const char* getLastError() const {
        return lastError_.c_str();
    }

private:
    const ParamSummary* getParam(int entryIndex, int paramIndex) const {
        if (entryIndex < 0 || static_cast<size_t>(entryIndex) >= entries_.size()) {
            return nullptr;
        }

        const EntrySummary& entry = entries_[static_cast<size_t>(entryIndex)];
        if (paramIndex < 0 || static_cast<size_t>(paramIndex) >= entry.params.size()) {
            return nullptr;
        }

        return &entry.params[static_cast<size_t>(paramIndex)];
    }

    void clearProgramState() {
        loadedPath_.clear();
        entries_.clear();
        lastResult_.clear();
        lastUInt32Result_.clear();
        lastError_.clear();
        scratch_.clear();
    }

    std::unique_ptr<PTXVM> createVm() {
        auto vm = std::make_unique<PTXVM>();
        if (!vm->initialize()) {
            lastError_ = "Failed to initialize the PTX virtual machine.";
            return nullptr;
        }

        return vm;
    }

    const PTXFunction* resolveKernel(const PTXProgram& program, const std::string& requestedKernel) const {
        if (!requestedKernel.empty()) {
            return program.getEntryByName(requestedKernel);
        }

        return program.getMainEntry();
    }

    bool isVectorAddSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 4) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               kernel.parameters[2].isPointer &&
               !kernel.parameters[3].isPointer &&
               isScalar32BitInteger(kernel.parameters[3]);
    }

    bool isMatrixMultiplySignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 6) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               kernel.parameters[2].isPointer &&
               !kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               !kernel.parameters[5].isPointer &&
               isScalar32BitInteger(kernel.parameters[3]) &&
               isScalar32BitInteger(kernel.parameters[4]) &&
               isScalar32BitInteger(kernel.parameters[5]);
    }

    bool isMatrixAdditionSignature(const PTXFunction& kernel) const {
        return isVectorAddSignature(kernel);
    }

    bool isConvolution1DSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 5) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               kernel.parameters[2].isPointer &&
               !kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               isScalar32BitInteger(kernel.parameters[3]) &&
               isScalar32BitInteger(kernel.parameters[4]);
    }

    bool isConvolution2DSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 7) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               kernel.parameters[2].isPointer &&
               !kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               !kernel.parameters[5].isPointer &&
               !kernel.parameters[6].isPointer &&
               isScalar32BitInteger(kernel.parameters[3]) &&
               isScalar32BitInteger(kernel.parameters[4]) &&
               isScalar32BitInteger(kernel.parameters[5]) &&
               isScalar32BitInteger(kernel.parameters[6]);
    }

    bool isConvolution3DSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 9) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               kernel.parameters[2].isPointer &&
               !kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               !kernel.parameters[5].isPointer &&
               !kernel.parameters[6].isPointer &&
               !kernel.parameters[7].isPointer &&
               !kernel.parameters[8].isPointer &&
               isScalar32BitInteger(kernel.parameters[3]) &&
               isScalar32BitInteger(kernel.parameters[4]) &&
               isScalar32BitInteger(kernel.parameters[5]) &&
               isScalar32BitInteger(kernel.parameters[6]) &&
               isScalar32BitInteger(kernel.parameters[7]) &&
               isScalar32BitInteger(kernel.parameters[8]);
    }

    bool isMatrixTransposeSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 4) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               !kernel.parameters[2].isPointer &&
               !kernel.parameters[3].isPointer &&
               isScalar32BitInteger(kernel.parameters[2]) &&
               isScalar32BitInteger(kernel.parameters[3]);
    }

    bool isReductionSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 3) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               !kernel.parameters[2].isPointer &&
               isScalar32BitInteger(kernel.parameters[2]);
    }

    bool isHistogrammingSignature(const PTXFunction& kernel) const {
        return isMatrixTransposeSignature(kernel);
    }

    bool isSoftmaxSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 3) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               !kernel.parameters[2].isPointer &&
               isScalar32BitInteger(kernel.parameters[2]);
    }

    bool isThreePointerTwoIntSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 5) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               kernel.parameters[2].isPointer &&
               !kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               isScalar32BitInteger(kernel.parameters[3]) &&
               isScalar32BitInteger(kernel.parameters[4]);
    }

    bool isTwoPointerTwoIntSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 4) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               !kernel.parameters[2].isPointer &&
               !kernel.parameters[3].isPointer &&
               isScalar32BitInteger(kernel.parameters[2]) &&
               isScalar32BitInteger(kernel.parameters[3]);
    }

    bool isThreePointerFourIntSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 7) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               kernel.parameters[2].isPointer &&
               !kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               !kernel.parameters[5].isPointer &&
               !kernel.parameters[6].isPointer &&
               isScalar32BitInteger(kernel.parameters[3]) &&
               isScalar32BitInteger(kernel.parameters[4]) &&
               isScalar32BitInteger(kernel.parameters[5]) &&
               isScalar32BitInteger(kernel.parameters[6]);
    }

    bool isThreePointerThreeIntSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 6) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               kernel.parameters[2].isPointer &&
               !kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               !kernel.parameters[5].isPointer &&
               isScalar32BitInteger(kernel.parameters[3]) &&
               isScalar32BitInteger(kernel.parameters[4]) &&
               isScalar32BitInteger(kernel.parameters[5]);
    }

    bool isThreePointerThreeIntThreeFloatThreeIntSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 12) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               kernel.parameters[2].isPointer &&
               !kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               !kernel.parameters[5].isPointer &&
               !kernel.parameters[6].isPointer &&
               !kernel.parameters[7].isPointer &&
               !kernel.parameters[8].isPointer &&
               !kernel.parameters[9].isPointer &&
               !kernel.parameters[10].isPointer &&
               !kernel.parameters[11].isPointer &&
               isScalar32BitInteger(kernel.parameters[3]) &&
               isScalar32BitInteger(kernel.parameters[4]) &&
               isScalar32BitInteger(kernel.parameters[5]) &&
               isScalar32BitFloat(kernel.parameters[6]) &&
               isScalar32BitFloat(kernel.parameters[7]) &&
               isScalar32BitFloat(kernel.parameters[8]) &&
               isScalar32BitInteger(kernel.parameters[9]) &&
               isScalar32BitInteger(kernel.parameters[10]) &&
               isScalar32BitInteger(kernel.parameters[11]);
    }

    bool isSevenPointerThreeIntSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 10) {
            return false;
        }

        for (size_t i = 0; i < 7; ++i) {
            if (!kernel.parameters[i].isPointer) {
                return false;
            }
        }

        return !kernel.parameters[7].isPointer &&
               !kernel.parameters[8].isPointer &&
               !kernel.parameters[9].isPointer &&
               isScalar32BitInteger(kernel.parameters[7]) &&
               isScalar32BitInteger(kernel.parameters[8]) &&
               isScalar32BitInteger(kernel.parameters[9]);
    }

    bool isTwoPointerSevenIntSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 9) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               !kernel.parameters[2].isPointer &&
               !kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               !kernel.parameters[5].isPointer &&
               !kernel.parameters[6].isPointer &&
               !kernel.parameters[7].isPointer &&
               !kernel.parameters[8].isPointer &&
               isScalar32BitInteger(kernel.parameters[2]) &&
               isScalar32BitInteger(kernel.parameters[3]) &&
               isScalar32BitInteger(kernel.parameters[4]) &&
               isScalar32BitInteger(kernel.parameters[5]) &&
               isScalar32BitInteger(kernel.parameters[6]) &&
               isScalar32BitInteger(kernel.parameters[7]) &&
               isScalar32BitInteger(kernel.parameters[8]);
    }

    bool isTwoPointerTwoFloatOneIntSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 5) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               !kernel.parameters[2].isPointer &&
               !kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               isScalar32BitFloat(kernel.parameters[2]) &&
               isScalar32BitFloat(kernel.parameters[3]) &&
               isScalar32BitInteger(kernel.parameters[4]);
    }

    bool isFourPointerTwoIntSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 6) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               kernel.parameters[2].isPointer &&
               kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               !kernel.parameters[5].isPointer &&
               isScalar32BitInteger(kernel.parameters[4]) &&
               isScalar32BitInteger(kernel.parameters[5]);
    }

    bool isFourPointerOneIntSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 5) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               kernel.parameters[2].isPointer &&
               kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               isScalar32BitInteger(kernel.parameters[4]);
    }

    bool isFourPointerThreeIntSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 7) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               kernel.parameters[2].isPointer &&
               kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               !kernel.parameters[5].isPointer &&
               !kernel.parameters[6].isPointer &&
               isScalar32BitInteger(kernel.parameters[4]) &&
               isScalar32BitInteger(kernel.parameters[5]) &&
               isScalar32BitInteger(kernel.parameters[6]);
    }

    bool isFourPointerThreeIntOneFloatSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 8) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               kernel.parameters[2].isPointer &&
               kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               !kernel.parameters[5].isPointer &&
               !kernel.parameters[6].isPointer &&
               !kernel.parameters[7].isPointer &&
               isScalar32BitInteger(kernel.parameters[4]) &&
               isScalar32BitInteger(kernel.parameters[5]) &&
               isScalar32BitInteger(kernel.parameters[6]) &&
               isScalar32BitFloat(kernel.parameters[7]);
    }

    bool isFourPointerTwoIntOneFloatSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 7) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               kernel.parameters[2].isPointer &&
               kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               !kernel.parameters[5].isPointer &&
               !kernel.parameters[6].isPointer &&
               isScalar32BitInteger(kernel.parameters[4]) &&
               isScalar32BitInteger(kernel.parameters[5]) &&
               isScalar32BitFloat(kernel.parameters[6]);
    }

    bool isSoftmaxAttentionSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 7) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               kernel.parameters[2].isPointer &&
               kernel.parameters[3].isPointer &&
               !kernel.parameters[4].isPointer &&
               !kernel.parameters[5].isPointer &&
               !kernel.parameters[6].isPointer &&
               isScalar32BitInteger(kernel.parameters[4]) &&
               isScalar32BitInteger(kernel.parameters[5]) &&
               isScalar32BitInteger(kernel.parameters[6]);
    }

    bool isMultiHeadAttentionSignature(const PTXFunction& kernel) const {
        return isSoftmaxAttentionSignature(kernel);
    }

    bool isColorInversionSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 3) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               !kernel.parameters[1].isPointer &&
               !kernel.parameters[2].isPointer &&
               isScalar32BitInteger(kernel.parameters[1]) &&
               isScalar32BitInteger(kernel.parameters[2]);
    }

    bool isInPlaceArraySignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 2) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               !kernel.parameters[1].isPointer &&
               isScalar32BitInteger(kernel.parameters[1]);
    }

    bool isIntScalarReduceSignature(const PTXFunction& kernel, int scalarCount) const {
        if (kernel.parameters.size() != static_cast<size_t>(2 + scalarCount)) {
            return false;
        }

        if (!kernel.parameters[0].isPointer || !kernel.parameters[1].isPointer) {
            return false;
        }

        for (int i = 0; i < scalarCount; ++i) {
            const PTXParameter& param = kernel.parameters[static_cast<size_t>(i + 2)];
            if (param.isPointer || !isScalar32BitInteger(param)) {
                return false;
            }
        }

        return true;
    }

    bool isSquareMatrixTransformSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 3) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               !kernel.parameters[2].isPointer &&
               isScalar32BitInteger(kernel.parameters[2]);
    }

    bool isScalar32BitInteger(const PTXParameter& param) const {
        return param.type == ".u32" || param.type == ".s32";
    }

    bool isScalar32BitFloat(const PTXParameter& param) const {
        return param.type == ".f32";
    }

    CUdeviceptr packFloat32(float value) const {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return static_cast<CUdeviceptr>(bits);
    }

    std::string loadedPath_;
    std::vector<EntrySummary> entries_;
    std::vector<float> lastResult_;
    std::vector<std::uint32_t> lastUInt32Result_;
    std::string lastError_;
    std::string scratch_;
};

WebPTXBridge& bridge() {
    static WebPTXBridge instance;
    return instance;
}

} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int ptxvm_load_program(const char* path) {
    if (path == nullptr) {
        return 0;
    }

    return bridge().loadProgram(path) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_get_entry_count() {
    return bridge().getEntryCount();
}

EMSCRIPTEN_KEEPALIVE const char* ptxvm_get_entry_name(int entryIndex) {
    return bridge().getEntryName(entryIndex);
}

EMSCRIPTEN_KEEPALIVE int ptxvm_get_param_count(int entryIndex) {
    return bridge().getParamCount(entryIndex);
}

EMSCRIPTEN_KEEPALIVE const char* ptxvm_get_param_name(int entryIndex, int paramIndex) {
    return bridge().getParamName(entryIndex, paramIndex);
}

EMSCRIPTEN_KEEPALIVE const char* ptxvm_get_param_type(int entryIndex, int paramIndex) {
    return bridge().getParamType(entryIndex, paramIndex);
}

EMSCRIPTEN_KEEPALIVE int ptxvm_is_param_pointer(int entryIndex, int paramIndex) {
    return bridge().isParamPointer(entryIndex, paramIndex);
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_vector_add(const char* kernelName,
                                              const float* inputA,
                                              int inputALen,
                                              const float* inputB,
                                              int inputBLen) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runVectorAddDemo(chosenKernel, inputA, inputALen, inputB, inputBLen) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_matrix_addition(const char* kernelName,
                                                   const float* inputA,
                                                   int inputALen,
                                                   const float* inputB,
                                                   int inputBLen,
                                                   int matrixN) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runMatrixAdditionDemo(
        chosenKernel,
        inputA,
        inputALen,
        inputB,
        inputBLen,
        matrixN) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_convolution_1d(const char* kernelName,
                                                  const float* input,
                                                  int inputLen,
                                                  const float* kernelInput,
                                                  int kernelLen) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runConvolution1DDemo(
        chosenKernel,
        input,
        inputLen,
        kernelInput,
        kernelLen) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_convolution_2d(const char* kernelName,
                                                  const float* input,
                                                  int inputLen,
                                                  const float* kernelInput,
                                                  int kernelLen,
                                                  int inputRows,
                                                  int inputCols,
                                                  int kernelRows,
                                                  int kernelCols) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runConvolution2DDemo(
        chosenKernel,
        input,
        inputLen,
        kernelInput,
        kernelLen,
        inputRows,
        inputCols,
        kernelRows,
        kernelCols) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_padded_convolution_2d(const char* kernelName,
                                                         const float* input,
                                                         int inputLen,
                                                         const float* kernelInput,
                                                         int kernelLen,
                                                         int inputRows,
                                                         int inputCols,
                                                         int kernelRows,
                                                         int kernelCols) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runPaddedConvolution2DDemo(
        chosenKernel,
        input,
        inputLen,
        kernelInput,
        kernelLen,
        inputRows,
        inputCols,
        kernelRows,
        kernelCols) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_convolution_3d(const char* kernelName,
                                                  const float* input,
                                                  int inputLen,
                                                  const float* kernelInput,
                                                  int kernelLen,
                                                  int inputDepth,
                                                  int inputRows,
                                                  int inputCols,
                                                  int kernelDepth,
                                                  int kernelRows,
                                                  int kernelCols) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runConvolution3DDemo(
        chosenKernel,
        input,
        inputLen,
        kernelInput,
        kernelLen,
        inputDepth,
        inputRows,
        inputCols,
        kernelDepth,
        kernelRows,
        kernelCols) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_sorting(const char* kernelName,
                                           float* input,
                                           int inputLen) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runSortingDemo(chosenKernel, input, inputLen) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_top_k(const char* kernelName,
                                         const float* input,
                                         int inputLen,
                                         int k) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runTopKDemo(chosenKernel, input, inputLen, k) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_top_p_sampling(const char* kernelName,
                                                  const float* logits,
                                                  int vocabSize,
                                                  const float* pValue,
                                                  const std::int32_t* seedValue) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runTopPSamplingDemo(
        chosenKernel,
        logits,
        vocabSize,
        pValue,
        seedValue) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_moe_top_k_gating(const char* kernelName,
                                                    const float* logits,
                                                    int logitsLen,
                                                    int rowsM,
                                                    int expertCount,
                                                    int topK) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runMoeTopKGatingDemo(
        chosenKernel,
        logits,
        logitsLen,
        rowsM,
        expertCount,
        topK) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_int_scalar_reduce(const char* kernelName,
                                                     const std::int32_t* input,
                                                     int inputLen,
                                                     int scalarCount,
                                                     const std::int32_t* scalars) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    std::vector<int> scalarValues;
    if (scalarCount > 0 && scalars != nullptr) {
        scalarValues.reserve(static_cast<size_t>(scalarCount));
        for (int i = 0; i < scalarCount; ++i) {
            scalarValues.push_back(static_cast<int>(scalars[i]));
        }
    }
    return bridge().runIntScalarReduceDemo(chosenKernel, input, inputLen, scalarValues) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_interleave(const char* kernelName,
                                              const float* inputA,
                                              int inputALen,
                                              const float* inputB,
                                              int inputBLen) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runInterleaveDemo(
        chosenKernel,
        inputA,
        inputALen,
        inputB,
        inputBLen) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_grayscale(const char* kernelName,
                                             const float* input,
                                             int inputLen,
                                             int width,
                                             int height) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runGrayscaleDemo(
        chosenKernel,
        input,
        inputLen,
        width,
        height) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_jacobi_2d(const char* kernelName,
                                             const float* input,
                                             int inputLen,
                                             int rows,
                                             int cols) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runJacobi2DDemo(
        chosenKernel,
        input,
        inputLen,
        rows,
        cols) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_merge_sorted(const char* kernelName,
                                                const float* inputA,
                                                int inputALen,
                                                const float* inputB,
                                                int inputBLen) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runMergeSortedDemo(
        chosenKernel,
        inputA,
        inputALen,
        inputB,
        inputBLen) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_matrix_multiplication(const char* kernelName,
                                                        const float* inputA,
                                                        int inputALen,
                                                        const float* inputB,
                                                        int inputBLen,
                                                         int rowsM,
                                                         int sharedN,
                                                         int colsK) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runMatrixMultiplicationDemo(
        chosenKernel,
        inputA,
        inputALen,
        inputB,
        inputBLen,
        rowsM,
        sharedN,
        colsK) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_int8_quantized_matmul(const char* kernelName,
                                                         const std::int8_t* inputA,
                                                         int inputALen,
                                                         const std::int8_t* inputB,
                                                         int inputBLen,
                                                         int rowsM,
                                                         int colsN,
                                                         int sharedK,
                                                         float scaleA,
                                                         float scaleB,
                                                         float scaleC,
                                                         int zeroPointA,
                                                         int zeroPointB,
                                                         int zeroPointC) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runInt8QuantizedMatMulDemo(
        chosenKernel,
        inputA,
        inputALen,
        inputB,
        inputBLen,
        rowsM,
        colsN,
        sharedK,
        scaleA,
        scaleB,
        scaleC,
        zeroPointA,
        zeroPointB,
        zeroPointC) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_sparse_matvec(const char* kernelName,
                                                 const float* inputA,
                                                 int inputALen,
                                                 const float* inputB,
                                                 int inputBLen,
                                                 int rowsM,
                                                 int colsN,
                                                 int nnz) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runSparseMatVecDemo(
        chosenKernel,
        inputA,
        inputALen,
        inputB,
        inputBLen,
        rowsM,
        colsN,
        nnz) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_batched_matrix_multiplication(const char* kernelName,
                                                                 const float* inputA,
                                                                 int inputALen,
                                                                 const float* inputB,
                                                                 int inputBLen,
                                                                 int batchSize,
                                                                 int rowsM,
                                                                 int sharedN,
                                                                 int colsK) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runBatchedMatrixMultiplicationDemo(
        chosenKernel,
        inputA,
        inputALen,
        inputB,
        inputBLen,
        batchSize,
        rowsM,
        sharedN,
        colsK) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_multi_agent_simulation(const char* kernelName,
                                                          const float* input,
                                                          int inputLen,
                                                          int agentCount) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runMultiAgentSimulationDemo(
        chosenKernel,
        input,
        inputLen,
        agentCount) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_nearest_neighbor(const char* kernelName,
                                                    const float* input,
                                                    int inputLen,
                                                    int pointCount) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runNearestNeighborDemo(
        chosenKernel,
        input,
        inputLen,
        pointCount) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_k_means(const char* kernelName,
                                           const float* dataX,
                                           const float* dataY,
                                           const float* initialCentroidX,
                                           const float* initialCentroidY,
                                           int sampleSize,
                                           int clusterCount,
                                           int maxIterations) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runKMeansDemo(
        chosenKernel,
        dataX,
        dataY,
        initialCentroidX,
        initialCentroidY,
        sampleSize,
        clusterCount,
        maxIterations) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_batch_normalization(const char* kernelName,
                                                       const float* input,
                                                       int inputLen,
                                                       const float* gamma,
                                                       int gammaLen,
                                                       const float* beta,
                                                       int betaLen,
                                                       int rowsN,
                                                       int colsC,
                                                       float eps) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runBatchNormalizationDemo(
        chosenKernel,
        input,
        inputLen,
        gamma,
        gammaLen,
        beta,
        betaLen,
        rowsN,
        colsC,
        eps) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_regression_vector(const char* kernelName,
                                                     const float* inputA,
                                                     int inputALen,
                                                     const float* inputB,
                                                     int inputBLen,
                                                     int sampleCount,
                                                     int featureCount) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runRegressionVectorDemo(
        chosenKernel,
        inputA,
        inputALen,
        inputB,
        inputBLen,
        sampleCount,
        featureCount) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_fft(const char* kernelName,
                                       const float* input,
                                       int inputLen,
                                       int signalLength) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runFftDemo(
        chosenKernel,
        input,
        inputLen,
        signalLength) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_max_pooling_2d(const char* kernelName,
                                                  const float* input,
                                                  int inputLen,
                                                  int batchSize,
                                                  int channelCount,
                                                  int height,
                                                  int width,
                                                  int kernelSize,
                                                  int stride,
                                                  int padding) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runMaxPooling2DDemo(
        chosenKernel,
        input,
        inputLen,
        batchSize,
        channelCount,
        height,
        width,
        kernelSize,
        stride,
        padding) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_matrix_transpose(const char* kernelName,
                                                    const float* inputA,
                                                    int inputALen,
                                                    int rows,
                                                    int cols) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runMatrixTransposeDemo(
        chosenKernel,
        inputA,
        inputALen,
        rows,
        cols) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_reduction(const char* kernelName,
                                             const float* input,
                                             int inputLen) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runReductionDemo(chosenKernel, input, inputLen) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_histogramming(const char* kernelName,
                                                 const std::int32_t* input,
                                                 int inputLen,
                                                 int numBins) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runHistogrammingDemo(chosenKernel, input, inputLen, numBins) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_cross_entropy(const char* kernelName,
                                                 const float* logits,
                                                 int logitsLen,
                                                 const std::int32_t* labels,
                                                 int labelsLen,
                                                 int rowsN,
                                                 int colsC) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runCrossEntropyDemo(
        chosenKernel,
        logits,
        logitsLen,
        labels,
        labelsLen,
        rowsN,
        colsC) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_monte_carlo(const char* kernelName,
                                               const float* samples,
                                               int sampleCount,
                                               float lowerBound,
                                               float upperBound) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runMonteCarloDemo(
        chosenKernel,
        samples,
        sampleCount,
        lowerBound,
        upperBound) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_rms_normalization(const char* kernelName,
                                                     const float* input,
                                                     int inputLen,
                                                     float gamma,
                                                     float beta) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runRmsNormalizationDemo(
        chosenKernel,
        input,
        inputLen,
        gamma,
        beta) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_softmax(const char* kernelName,
                                           const float* input,
                                           int inputLen) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runSoftmaxDemo(chosenKernel, input, inputLen) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_softmax_attention(const char* kernelName,
                                                     const float* inputQ,
                                                     int inputQLen,
                                                     const float* inputK,
                                                     int inputKLen,
                                                     const float* inputV,
                                                     int inputVLen,
                                                     int rowsM,
                                                     int sharedN,
                                                     int featureD) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runSoftmaxAttentionDemo(
        chosenKernel,
        inputQ,
        inputQLen,
        inputK,
        inputKLen,
        inputV,
        inputVLen,
        rowsM,
        sharedN,
        featureD) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_linear_self_attention(const char* kernelName,
                                                         const float* inputQ,
                                                         int inputQLen,
                                                         const float* inputK,
                                                         int inputKLen,
                                                         const float* inputV,
                                                         int inputVLen,
                                                         int rowsM,
                                                         int featureD) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runLinearSelfAttentionDemo(
        chosenKernel,
        inputQ,
        inputQLen,
        inputK,
        inputKLen,
        inputV,
        inputVLen,
        rowsM,
        featureD) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_rotary_embedding(const char* kernelName,
                                                    const float* inputQ,
                                                    int inputQLen,
                                                    const float* inputCos,
                                                    int inputCosLen,
                                                    const float* inputSin,
                                                    int inputSinLen,
                                                    int rowsM,
                                                    int featureD) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runRotaryEmbeddingDemo(
        chosenKernel,
        inputQ,
        inputQLen,
        inputCos,
        inputCosLen,
        inputSin,
        inputSinLen,
        rowsM,
        featureD) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_multi_head_attention(const char* kernelName,
                                                        const float* inputQ,
                                                        int inputQLen,
                                                        const float* inputK,
                                                        int inputKLen,
                                                        const float* inputV,
                                                        int inputVLen,
                                                        int sequenceLen,
                                                        int modelDim,
                                                        int headCount) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runMultiHeadAttentionDemo(
        chosenKernel,
        inputQ,
        inputQLen,
        inputK,
        inputKLen,
        inputV,
        inputVLen,
        sequenceLen,
        modelDim,
        headCount) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_causal_self_attention(const char* kernelName,
                                                         const float* inputQ,
                                                         int inputQLen,
                                                         const float* inputK,
                                                         int inputKLen,
                                                         const float* inputV,
                                                         int inputVLen,
                                                         int rowsM,
                                                         int featureD) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runCausalSelfAttentionDemo(
        chosenKernel,
        inputQ,
        inputQLen,
        inputK,
        inputKLen,
        inputV,
        inputVLen,
        rowsM,
        featureD) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_sliding_window_attention(const char* kernelName,
                                                            const float* inputQ,
                                                            int inputQLen,
                                                            const float* inputK,
                                                            int inputKLen,
                                                            const float* inputV,
                                                            int inputVLen,
                                                            int rowsM,
                                                            int featureD,
                                                            int windowSize) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runSlidingWindowAttentionDemo(
        chosenKernel,
        inputQ,
        inputQLen,
        inputK,
        inputKLen,
        inputV,
        inputVLen,
        rowsM,
        featureD,
        windowSize) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_alibi_attention(const char* kernelName,
                                                   const float* inputQ,
                                                   int inputQLen,
                                                   const float* inputK,
                                                   int inputKLen,
                                                   const float* inputV,
                                                   int inputVLen,
                                                   int rowsM,
                                                   int sharedN,
                                                   int featureD,
                                                   float alpha) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runAlibiAttentionDemo(
        chosenKernel,
        inputQ,
        inputQLen,
        inputK,
        inputKLen,
        inputV,
        inputVLen,
        rowsM,
        sharedN,
        featureD,
        alpha) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_value_clipping(const char* kernelName,
                                                  const float* input,
                                                  int inputLen,
                                                  float lowerBound,
                                                  float upperBound) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runValueClippingDemo(
        chosenKernel,
        input,
        inputLen,
        lowerBound,
        upperBound) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_weight_dequantization(const char* kernelName,
                                                         const float* inputA,
                                                         int inputALen,
                                                         const float* inputB,
                                                         int inputBLen,
                                                         int rowsM,
                                                         int colsN,
                                                         int tileSize) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runWeightDequantizationDemo(
        chosenKernel,
        inputA,
        inputALen,
        inputB,
        inputBLen,
        rowsM,
        colsN,
        tileSize) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_color_inversion(const char* kernelName,
                                                   const std::uint8_t* image,
                                                   int imageLen,
                                                   int width,
                                                   int height) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runColorInversionDemo(
        chosenKernel,
        image,
        imageLen,
        width,
        height) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_elementwise(const char* kernelName,
                                               const float* input,
                                               int inputLen) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runElementwiseDemo(chosenKernel, input, inputLen) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_elementwise_int(const char* kernelName,
                                                    const std::int32_t* input,
                                                    int inputLen) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runElementwiseIntDemo(chosenKernel, input, inputLen) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_int_array_transform(const char* kernelName,
                                                       const std::int32_t* input,
                                                       int inputLen) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runIntArrayTransformDemo(chosenKernel, input, inputLen) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_rainbow_table(const char* kernelName,
                                                 const std::int32_t* input,
                                                 int inputLen,
                                                 int rounds) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runRainbowTableDemo(chosenKernel, input, inputLen, rounds) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_matrix_copy(const char* kernelName,
                                                const float* input,
                                                int inputLen) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runMatrixCopyDemo(chosenKernel, input, inputLen) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_matrix_power(const char* kernelName,
                                                const float* input,
                                                int inputLen,
                                                int matrixSize,
                                                int power) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runMatrixPowerDemo(chosenKernel, input, inputLen, matrixSize, power) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_square_matrix_transform(const char* kernelName,
                                                           const float* input,
                                                           int inputLen,
                                                           int matrixSize) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runSquareMatrixTransformDemo(chosenKernel, input, inputLen, matrixSize) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_dot_product(const char* kernelName,
                                                const float* inputA,
                                                int inputALen,
                                                const float* inputB,
                                                int inputBLen) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runDotProductDemo(chosenKernel, inputA, inputALen, inputB, inputBLen) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_prefix_sum(const char* kernelName,
                                               const float* input,
                                               int inputLen) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runPrefixSumDemo(chosenKernel, input, inputLen) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_run_reverse_array(const char* kernelName,
                                                  float* input,
                                                  int inputLen) {
    const std::string chosenKernel = kernelName == nullptr ? "" : kernelName;
    return bridge().runReverseArrayDemo(chosenKernel, input, inputLen) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int ptxvm_get_result_count() {
    return bridge().getResultCount();
}

EMSCRIPTEN_KEEPALIVE double ptxvm_get_result_value(int index) {
    return bridge().getResultValue(index);
}

EMSCRIPTEN_KEEPALIVE const char* ptxvm_get_last_error() {
    return bridge().getLastError();
}

} // extern "C"
