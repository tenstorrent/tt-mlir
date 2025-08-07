// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_SFPI_IR_SFPIOPSATTRS_TD
#define TTMLIR_DIALECT_SFPI_IR_SFPIOPSATTRS_TD

include "ttmlir/Dialect/SFPI/IR/SFPIBase.td"
include "ttmlir/Dialect/SFPI/IR/SFPIEnums.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// SFPI enum attribute definitions
//===----------------------------------------------------------------------===//

def SFPI_LoadMod0Attr : EnumAttr<SFPI_Dialect, SFPI_LOAD_MOD0, "load_mod0">;
def SFPI_LoadAddrModeAttr
    : EnumAttr<SFPI_Dialect, SFPI_LOAD_ADDR_MODE, "load_addr_mode">;
def SFPI_StoreMod0Attr : EnumAttr<SFPI_Dialect, SFPI_STORE_MOD0, "store_mod0">;
def SFPI_StoreAddrModeAttr
    : EnumAttr<SFPI_Dialect, SFPI_STORE_ADDR_MODE, "store_addr_mode">;
def SFPI_MovMod1Attr : EnumAttr<SFPI_Dialect, SFPI_MOV_MOD1, "mov_mod1">;
def SFPI_MadMod1Attr : EnumAttr<SFPI_Dialect, SFPI_MAD_MOD1, "mad_mod1">;
def SFPI_LoadIMod0Attr : EnumAttr<SFPI_Dialect, SFPI_LOADI_MOD0, "loadi_mod0">;
def SFPI_XLoadIMod0Attr
    : EnumAttr<SFPI_Dialect, SFPI_XLOADI_MOD0, "xloadi_mod0">;
def SFPI_ExManMod1Attr : EnumAttr<SFPI_Dialect, SFPI_EXMAN_MOD1, "exman_mod1">;
def SFPI_ExExpMod1Attr : EnumAttr<SFPI_Dialect, SFPI_EXEXP_MOD1, "exexp_mod1">;
def SFPI_AbsMod1Attr : EnumAttr<SFPI_Dialect, SFPI_ABS_MOD1, "abs_mod1">;
def SFPI_AddMod1Attr : EnumAttr<SFPI_Dialect, SFPI_ADD_MOD1, "add_mod1">;
def SFPI_XiAddMod1Attr : EnumAttr<SFPI_Dialect, SFPI_XIADD_MOD1, "xiadd_mod1">;
def SFPI_XCmpMod1Attr : EnumAttr<SFPI_Dialect, SFPI_XCMP_MOD1, "xcmp_mod1">;
def SFPI_XSCmpMod1Attr : EnumAttr<SFPI_Dialect, SFPI_XSCMP_MOD1, "xscmp_mod1">;
def SFPI_XBoolMod1Attr : EnumAttr<SFPI_Dialect, SFPI_XBOOL_MOD1, "xbool_mod1">;
def SFPI_SetCCMod1Attr : EnumAttr<SFPI_Dialect, SFPI_SETCC_MOD1, "setcc_mod1">;
def SFPI_EnCCImm12Attr : EnumAttr<SFPI_Dialect, SFPI_ENCC_IMM12, "encc_imm12">;
def SFPI_EnCCMod1Attr : EnumAttr<SFPI_Dialect, SFPI_ENCC_MOD1, "encc_mod1">;
def SFPI_PushCMod1Attr : EnumAttr<SFPI_Dialect, SFPI_PUSHC_MOD1, "pushc_mod1">;
def SFPI_LzMod1Attr : EnumAttr<SFPI_Dialect, SFPI_LZ_MOD1, "lz_mod1">;
def SFPI_SDivP2Mod1Attr
    : EnumAttr<SFPI_Dialect, SFPI_SDIVP2_MOD1, "sdivp2_mod1">;
def SFPI_LutMod0Attr : EnumAttr<SFPI_Dialect, SFPI_LUT_MOD0, "lut_mod0">;
def SFPI_LutFp32Mod0Attr
    : EnumAttr<SFPI_Dialect, SFPI_LUTFP32_MOD0, "lutfp32_mod0">;
def SFPI_CastMod1Attr : EnumAttr<SFPI_Dialect, SFPI_CAST_MOD1, "cast_mod1">;
def SFPI_StochRndRndAttr
    : EnumAttr<SFPI_Dialect, SFPI_STOCHRND_RND, "stoch_rnd_rnd">;
def SFPI_StochRndMod1Attr
    : EnumAttr<SFPI_Dialect, SFPI_STOCHRND_MOD1, "stoch_rnd_mod1">;
def SFPI_Shft2Mod1Attr : EnumAttr<SFPI_Dialect, SFPI_SHFT2_MOD1, "shft2_mod1">;
def SFPI_SwapMod1Attr : EnumAttr<SFPI_Dialect, SFPI_SWAP_MOD1, "swap_mod1">;
def SFPI_ConfigDestAttr
    : EnumAttr<SFPI_Dialect, SFPI_CONFIG_DEST, "config_dest">;
def SFPI_ConfigMod1Attr
    : EnumAttr<SFPI_Dialect, SFPI_CONFIG_MOD1, "config_mod1">;
def SFPI_CRegIdxAttr : EnumAttr<SFPI_Dialect, SFPI_CREG_IDX, "creg_idx">;

#endif // TTMLIR_DIALECT_SFPI_IR_SFPIOPSATTRS_TD
