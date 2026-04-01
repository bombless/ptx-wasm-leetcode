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

    int getResultCount() const {
        return static_cast<int>(lastResult_.size());
    }

    double getResultValue(int index) const {
        if (index < 0 || static_cast<size_t>(index) >= lastResult_.size()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return static_cast<double>(lastResult_[static_cast<size_t>(index)]);
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

    bool isSoftmaxSignature(const PTXFunction& kernel) const {
        if (kernel.parameters.size() != 3) {
            return false;
        }

        return kernel.parameters[0].isPointer &&
               kernel.parameters[1].isPointer &&
               !kernel.parameters[2].isPointer &&
               isScalar32BitInteger(kernel.parameters[2]);
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

    bool isScalar32BitInteger(const PTXParameter& param) const {
        return param.type == ".u32" || param.type == ".s32";
    }

    std::string loadedPath_;
    std::vector<EntrySummary> entries_;
    std::vector<float> lastResult_;
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
