#ifndef REGISTER_BANK_HPP
#define REGISTER_BANK_HPP

#include <cstdint>
#include <vector>
#include <cstring>

// 特殊寄存器枚举
enum class SpecialRegister {
    // 线程标识
    TID_X, TID_Y, TID_Z,
    // 线程块大小
    NTID_X, NTID_Y, NTID_Z,
    // 线程块标识
    CTAID_X, CTAID_Y, CTAID_Z,
    // 网格大小
    NCTAID_X, NCTAID_Y, NCTAID_Z,
    // Warp 大小
    WARPSIZE,
    // Lane ID
    LANEID,
    // 时钟
    CLOCK, CLOCK64
};

class RegisterBank {
public:
    // Constructor/destructor
    RegisterBank();
    ~RegisterBank();

    // Initialize register bank with specified number of registers
    bool initialize(size_t numRegisters = 32, size_t numFloatRegisters = 32);

    // 🔧 整数寄存器操作 (%r0-%rN)
    uint64_t readRegister(size_t registerIndex) const;
    void writeRegister(size_t registerIndex, uint64_t value);

    // 🔧 浮点寄存器操作 (%f0-%fN)
    float readFloatRegister(size_t registerIndex) const;
    void writeFloatRegister(size_t registerIndex, float value);

    uint16_t readHalfRegisterBits(size_t registerIndex) const;
    void writeHalfRegisterBits(size_t registerIndex, uint16_t value);
    
    double readDoubleRegister(size_t registerIndex) const;
    void writeDoubleRegister(size_t registerIndex, double value);

    // 谓词寄存器操作 (%p0-%p7)
    bool readPredicate(size_t predicateIndex) const;
    void writePredicate(size_t predicateIndex, bool value);

    // 🔧 特殊寄存器操作
    uint32_t readSpecialRegister(SpecialRegister reg) const;
    void setThreadId(uint32_t x, uint32_t y, uint32_t z);
    void setBlockId(uint32_t x, uint32_t y, uint32_t z);
    void setThreadDimensions(uint32_t x, uint32_t y, uint32_t z);
    void setGridDimensions(uint32_t x, uint32_t y, uint32_t z);
    void setWarpSize(uint32_t size);
    void setLaneId(uint32_t id);

    // Get number of registers
    size_t getNumRegisters() const;
    size_t getNumFloatRegisters() const;
    size_t getNumPredicateRegisters() const;

private:
    // 整数寄存器 (%r0-%rN, %rd0-%rdN)
    std::vector<uint64_t> m_registers;
    
    // 浮点寄存器 (%f0-%fN for float, %fd0-%fdN for double)
    std::vector<uint64_t> m_floatRegisters;  // 使用 uint64_t 存储，支持 float 和 double
    
    // 谓词寄存器 (%p0-%p7)
    std::vector<bool> m_predicateRegisters;
    
    // 特殊寄存器
    struct {
        uint32_t tid_x, tid_y, tid_z;        // 线程ID
        uint32_t ntid_x, ntid_y, ntid_z;     // 块大小
        uint32_t ctaid_x, ctaid_y, ctaid_z;  // 块ID
        uint32_t nctaid_x, nctaid_y, nctaid_z; // 网格大小
        uint32_t warpsize;                    // Warp大小
        uint32_t laneid;                      // Lane ID
        uint64_t clock;                       // 时钟周期
    } m_specialRegs;
    
    // Configuration
    size_t m_numRegisters;
    size_t m_numFloatRegisters;
    size_t m_numPredicateRegisters;
};

#endif // REGISTER_BANK_HPP
