#include <gtest/gtest.h>

#include "parser.hpp"

TEST(ParserCompatibilityTest, ParsesVisibleEntryWithHostSignature)
{
    const std::string testPTX = R"(
.version 8.7
.target sm_75
.address_size 64

.visible .entry vector_add(float const*, float const*, float*, int)(
    .param .u64 vector_add_param_0,
    .param .u64 vector_add_param_1,
    .param .u64 vector_add_param_2,
    .param .u32 vector_add_param_3
)
{
    ret;
}
)";

    PTXParser parser;
    ASSERT_TRUE(parser.parseString(testPTX)) << parser.getErrorMessage();

    const PTXProgram& program = parser.getProgram();
    ASSERT_EQ(program.functions.size(), 1u);
    ASSERT_EQ(program.entryPoints.size(), 1u);

    const PTXFunction& entry = program.functions.front();
    EXPECT_TRUE(entry.isEntry);
    EXPECT_EQ(entry.name, "vector_add");
    ASSERT_EQ(entry.parameters.size(), 4u);

    EXPECT_EQ(entry.parameters[0].type, ".u64");
    EXPECT_TRUE(entry.parameters[0].isPointer);
    EXPECT_EQ(entry.parameters[1].type, ".u64");
    EXPECT_TRUE(entry.parameters[1].isPointer);
    EXPECT_EQ(entry.parameters[2].type, ".u64");
    EXPECT_TRUE(entry.parameters[2].isPointer);
    EXPECT_EQ(entry.parameters[3].type, ".u32");
    EXPECT_FALSE(entry.parameters[3].isPointer);
}

TEST(ParserCompatibilityTest, ParsesSpecialRegistersAsDedicatedOperands)
{
    const std::string testPTX = R"(
.version 8.0
.target sm_89
.address_size 64

.entry solve(
    .param .u64 out_ptr
)
{
    .reg .u32 %r<4>;

$BB0:
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.y;
    mov.u32 %r3, %ntid.x;
    ret;
}
)";

    PTXParser parser;
    ASSERT_TRUE(parser.parseString(testPTX)) << parser.getErrorMessage();

    const auto& instructions = parser.getInstructions();
    ASSERT_GE(instructions.size(), 3u);

    EXPECT_EQ(instructions[0].sources[0].type, OperandType::SPECIAL);
    EXPECT_EQ(instructions[0].sources[0].labelName, "%tid.x");

    EXPECT_EQ(instructions[1].sources[0].type, OperandType::SPECIAL);
    EXPECT_EQ(instructions[1].sources[0].labelName, "%ctaid.y");

    EXPECT_EQ(instructions[2].sources[0].type, OperandType::SPECIAL);
    EXPECT_EQ(instructions[2].sources[0].labelName, "%ntid.x");
}
