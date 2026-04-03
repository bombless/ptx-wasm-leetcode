#include "register_bank.hpp"
#include <stdexcept>
#include <cstring>

RegisterBank::RegisterBank() : 
    m_numRegisters(0),
    m_numFloatRegisters(0),
    m_numPredicateRegisters(0) {
    // 初始化特殊寄存器
    std::memset(&m_specialRegs, 0, sizeof(m_specialRegs));
    m_specialRegs.warpsize = 32;  // NVIDIA GPU 默认 warp 大小
}

RegisterBank::~RegisterBank() {
    // Default destructor
}

bool RegisterBank::initialize(size_t numRegisters, size_t numFloatRegisters) {
    try {
        // 初始化整数寄存器
        m_registers.resize(numRegisters, 0);
        m_numRegisters = numRegisters;
        
        // 🔧 初始化浮点寄存器
        m_floatRegisters.resize(numFloatRegisters, 0);
        m_numFloatRegisters = numFloatRegisters;
        
        // Allocate a larger predicate file so browser-side kernels with
        // compiler-generated control flow do not run out of %p registers.
        m_predicateRegisters.resize(128, false);
        m_numPredicateRegisters = 128;
        
        return true;
    } catch (const std::bad_alloc&) {
        return false;
    }
}

// 整数寄存器操作
uint64_t RegisterBank::readRegister(size_t registerIndex) const {
    if (registerIndex >= m_numRegisters) {
        throw std::out_of_range("Register index out of range");
    }
    return m_registers[registerIndex];
}

void RegisterBank::writeRegister(size_t registerIndex, uint64_t value) {
    if (registerIndex >= m_numRegisters) {
        throw std::out_of_range("Register index out of range");
    }
    m_registers[registerIndex] = value;
}

// 🔧 浮点寄存器操作
float RegisterBank::readFloatRegister(size_t registerIndex) const {
    if (registerIndex >= m_numFloatRegisters) {
        throw std::out_of_range("Float register index out of range");
    }
    
    // 从 uint64_t 转换为 float
    uint32_t bits = static_cast<uint32_t>(m_floatRegisters[registerIndex] & 0xFFFFFFFF);
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

void RegisterBank::writeFloatRegister(size_t registerIndex, float value) {
    if (registerIndex >= m_numFloatRegisters) {
        throw std::out_of_range("Float register index out of range");
    }
    
    // 从 float 转换为 uint64_t
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(float));
    m_floatRegisters[registerIndex] = static_cast<uint64_t>(bits);
}

uint16_t RegisterBank::readHalfRegisterBits(size_t registerIndex) const {
    if (registerIndex >= m_numFloatRegisters) {
        throw std::out_of_range("Half register index out of range");
    }

    return static_cast<uint16_t>(m_floatRegisters[registerIndex] & 0xFFFF);
}

void RegisterBank::writeHalfRegisterBits(size_t registerIndex, uint16_t value) {
    if (registerIndex >= m_numFloatRegisters) {
        throw std::out_of_range("Half register index out of range");
    }

    m_floatRegisters[registerIndex] = static_cast<uint64_t>(value);
}

double RegisterBank::readDoubleRegister(size_t registerIndex) const {
    if (registerIndex >= m_numFloatRegisters) {
        throw std::out_of_range("Double register index out of range");
    }
    
    // 从 uint64_t 转换为 double
    double result;
    std::memcpy(&result, &m_floatRegisters[registerIndex], sizeof(double));
    return result;
}

void RegisterBank::writeDoubleRegister(size_t registerIndex, double value) {
    if (registerIndex >= m_numFloatRegisters) {
        throw std::out_of_range("Double register index out of range");
    }
    
    // 从 double 转换为 uint64_t
    std::memcpy(&m_floatRegisters[registerIndex], &value, sizeof(double));
}

// 谓词寄存器操作
bool RegisterBank::readPredicate(size_t predicateIndex) const {
    if (predicateIndex >= m_numPredicateRegisters) {
        throw std::out_of_range("Predicate register index out of range");
    }
    return m_predicateRegisters[predicateIndex];
}

void RegisterBank::writePredicate(size_t predicateIndex, bool value) {
    if (predicateIndex >= m_numPredicateRegisters) {
        throw std::out_of_range("Predicate register index out of range");
    }
    m_predicateRegisters[predicateIndex] = value;
}

// 🔧 特殊寄存器操作
uint32_t RegisterBank::readSpecialRegister(SpecialRegister reg) const {
    switch (reg) {
        case SpecialRegister::TID_X:    return m_specialRegs.tid_x;
        case SpecialRegister::TID_Y:    return m_specialRegs.tid_y;
        case SpecialRegister::TID_Z:    return m_specialRegs.tid_z;
        case SpecialRegister::NTID_X:   return m_specialRegs.ntid_x;
        case SpecialRegister::NTID_Y:   return m_specialRegs.ntid_y;
        case SpecialRegister::NTID_Z:   return m_specialRegs.ntid_z;
        case SpecialRegister::CTAID_X:  return m_specialRegs.ctaid_x;
        case SpecialRegister::CTAID_Y:  return m_specialRegs.ctaid_y;
        case SpecialRegister::CTAID_Z:  return m_specialRegs.ctaid_z;
        case SpecialRegister::NCTAID_X: return m_specialRegs.nctaid_x;
        case SpecialRegister::NCTAID_Y: return m_specialRegs.nctaid_y;
        case SpecialRegister::NCTAID_Z: return m_specialRegs.nctaid_z;
        case SpecialRegister::WARPSIZE: return m_specialRegs.warpsize;
        case SpecialRegister::LANEID:   return m_specialRegs.laneid;
        case SpecialRegister::CLOCK:    return static_cast<uint32_t>(m_specialRegs.clock & 0xFFFFFFFF);
        case SpecialRegister::CLOCK64:  return static_cast<uint32_t>(m_specialRegs.clock);
        default:
            throw std::invalid_argument("Unknown special register");
    }
}

void RegisterBank::setThreadId(uint32_t x, uint32_t y, uint32_t z) {
    m_specialRegs.tid_x = x;
    m_specialRegs.tid_y = y;
    m_specialRegs.tid_z = z;
    
    // 计算 lane ID（在 warp 内的位置）
    m_specialRegs.laneid = x % m_specialRegs.warpsize;
}

void RegisterBank::setBlockId(uint32_t x, uint32_t y, uint32_t z) {
    m_specialRegs.ctaid_x = x;
    m_specialRegs.ctaid_y = y;
    m_specialRegs.ctaid_z = z;
}

void RegisterBank::setThreadDimensions(uint32_t x, uint32_t y, uint32_t z) {
    m_specialRegs.ntid_x = x;
    m_specialRegs.ntid_y = y;
    m_specialRegs.ntid_z = z;
}

void RegisterBank::setGridDimensions(uint32_t x, uint32_t y, uint32_t z) {
    m_specialRegs.nctaid_x = x;
    m_specialRegs.nctaid_y = y;
    m_specialRegs.nctaid_z = z;
}

void RegisterBank::setWarpSize(uint32_t size) {
    m_specialRegs.warpsize = size;
}

void RegisterBank::setLaneId(uint32_t id) {
    m_specialRegs.laneid = id;
}

// Getters
size_t RegisterBank::getNumRegisters() const {
    return m_numRegisters;
}

size_t RegisterBank::getNumFloatRegisters() const {
    return m_numFloatRegisters;
}

size_t RegisterBank::getNumPredicateRegisters() const {
    return m_numPredicateRegisters;
}
