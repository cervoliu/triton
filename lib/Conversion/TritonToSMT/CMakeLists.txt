add_triton_library(TritonToSMT
    TritonToSMTPass.cpp

    DEPENDS
    TritonToSMTConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRArithDialect
    MLIRMathDialect
    MLIRSMT
    TritonIR
)
