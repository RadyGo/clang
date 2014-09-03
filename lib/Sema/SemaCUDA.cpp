//===--- SemaCUDA.cpp - Semantic Analysis for CUDA constructs -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file implements semantic analysis for CUDA constructs.
///
//===----------------------------------------------------------------------===//

#include "clang/Sema/Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Sema/SemaDiagnostic.h"
using namespace clang;

ExprResult Sema::ActOnCUDAExecConfigExpr(Scope *S, SourceLocation LLLLoc,
                                         MultiExprArg ExecConfig,
                                         SourceLocation GGGLoc) {
  FunctionDecl *ConfigDecl = Context.getcudaConfigureCallDecl();
  if (!ConfigDecl)
    return ExprError(Diag(LLLLoc, diag::err_undeclared_var_use)
                     << "cudaConfigureCall");
  QualType ConfigQTy = ConfigDecl->getType();

  DeclRefExpr *ConfigDR = new (Context)
      DeclRefExpr(ConfigDecl, false, ConfigQTy, VK_LValue, LLLLoc);
  MarkFunctionReferenced(LLLLoc, ConfigDecl);

  return ActOnCallExpr(S, ConfigDR, LLLLoc, ExecConfig, GGGLoc, nullptr,
                       /*IsExecConfig=*/true);
}

/// IdentifyCUDATarget - Determine the CUDA compilation target for this function
Sema::CUDAFunctionTarget Sema::IdentifyCUDATarget(const FunctionDecl *D) {
  // Implicitly declared functions (e.g. copy constructors) are
  // __host__ __device__
  //if (D->isImplicit()) {
    //return CFT_HostDevice;
  //}

  if (D->hasAttr<CUDAGlobalAttr>())
    return CFT_Global;

  if (D->hasAttr<CUDADeviceAttr>()) {
    if (D->hasAttr<CUDAHostAttr>())
      return CFT_HostDevice;
    return CFT_Device;
  }

  return CFT_Host;
}

bool Sema::CheckCUDATarget(const FunctionDecl *Caller,
                           const FunctionDecl *Callee) {
  llvm::errs() << "@@ CheckCUDATarget\n";
  llvm::errs() << "   Caller:\n";
  Caller->dump();
  llvm::errs() << "   Callee:\n";
  Callee->dump();

  return CheckCUDATarget(IdentifyCUDATarget(Caller),
                         IdentifyCUDATarget(Callee));
}

bool Sema::CheckCUDATarget(CUDAFunctionTarget CallerTarget,
                           CUDAFunctionTarget CalleeTarget) {
  // CUDA B.1.1 "The __device__ qualifier declares a function that is...
  // Callable from the device only."
  if (CallerTarget == CFT_Host && CalleeTarget == CFT_Device)
    return true;

  // CUDA B.1.2 "The __global__ qualifier declares a function that is...
  // Callable from the host only."
  // CUDA B.1.3 "The __host__ qualifier declares a function that is...
  // Callable from the host only."
  if ((CallerTarget == CFT_Device || CallerTarget == CFT_Global) &&
      (CalleeTarget == CFT_Host || CalleeTarget == CFT_Global))
    return true;

  if (CallerTarget == CFT_HostDevice && CalleeTarget != CFT_HostDevice)
    return true;

  return false;
}

static bool
resolveCalleeCUDATargetConflict(Sema::CUDAFunctionTarget Target1,
                                Sema::CUDAFunctionTarget Target2,
                                Sema::CUDAFunctionTarget *ResolvedTarget) {
  assert((Target1 != Sema::CFT_Global && Target2 != Sema::CFT_Global) &&
         "Special members cannot be marked global");

  if (Target1 == Sema::CFT_HostDevice) {
    *ResolvedTarget = Target2;
  } else if (Target2 == Sema::CFT_HostDevice) {
    *ResolvedTarget = Target1;
  } else if (Target1 != Target2) {
    return true;
  } else {
    *ResolvedTarget = Target1;
  }

  return true;
}

void Sema::inferCUDATargetForDefaultedSpecialMember(
    CXXRecordDecl *ClassDecl, CXXSpecialMember CSM,
    CXXConstructorDecl *CtorDecl, bool ConstRHS) {

  CUDAFunctionTarget InferredTarget;
  bool HasInferredTarget = false;

  for (const auto &B : ClassDecl->bases()) {
    const RecordType *BaseType = B.getType()->getAs<RecordType>();
    if (!BaseType) {
      continue;
    }

    CXXRecordDecl *BaseClassDecl = cast<CXXRecordDecl>(BaseType->getDecl());
    Sema::SpecialMemberOverloadResult *SMOR =
        LookupSpecialMember(BaseClassDecl, CSM,
                            /* ConstArg */ ConstRHS,
                            /* VolatileArg */ false,
                            /* RValueThis */ false,
                            /* ConstThis */ false,
                            /* VolatileThis */ false);

    if (!SMOR || !SMOR->getMethod()) {
      continue;
    }

    CXXMethodDecl *BaseMethod = SMOR->getMethod();
    CUDAFunctionTarget BaseMethodTarget = IdentifyCUDATarget(BaseMethod);
    llvm::errs() << "@@ inferCUDATargetForDefaultedSpecialMember found base\n";
    BaseMethod->dump();

    if (!HasInferredTarget) {
      HasInferredTarget = true;
      InferredTarget = BaseMethodTarget;
    } else {
      bool ResolutionError = resolveCalleeCUDATargetConflict(
          InferredTarget, BaseMethodTarget, &InferredTarget);
      if (ResolutionError) {
        // TODO(eliben): proper diagnostic here
        llvm::errs() << "!!! BOO BAD COLLISION: " << InferredTarget << " and "
                     << BaseMethodTarget << "\n";
        return;
      }
    }
  }

  for (const auto *F : ClassDecl->fields()) {
    if (F->isInvalidDecl()) {
      continue;
    }
    QualType QualBaseType = Context.getBaseElementType(F->getType());
    const RecordType *BaseType = QualBaseType->getAs<RecordType>();
    if (!BaseType) {
      continue;
    }

    CXXRecordDecl *FieldRecDecl = cast<CXXRecordDecl>(BaseType->getDecl());
    Sema::SpecialMemberOverloadResult *SMOR =
        LookupSpecialMember(FieldRecDecl, CSM,
                            /* ConstArg */ ConstRHS && !F->isMutable(),
                            /* VolatileArg */ false,
                            /* RValueThis */ false,
                            /* ConstThis */ false,
                            /* VolatileThis */ false);

    if (!SMOR || !SMOR->getMethod()) {
      continue;
    }

    CXXMethodDecl *FieldMethod = SMOR->getMethod();
    CUDAFunctionTarget FieldMethodTarget = IdentifyCUDATarget(FieldMethod);
    llvm::errs() << "@@ inferCUDATargetForDefaultedSpecialMember found field\n";
    FieldMethod->dump();

    if (!HasInferredTarget) {
      HasInferredTarget = true;
      InferredTarget = FieldMethodTarget;
    } else {
      bool ResolutionError = resolveCalleeCUDATargetConflict(
          InferredTarget, FieldMethodTarget, &InferredTarget);
      if (ResolutionError) {
        // TODO(eliben): proper diagnostic here
        llvm::errs() << "!!! BOO BAD COLLISION: " << InferredTarget << " and "
                     << FieldMethodTarget << "\n";
        return;
      }
    }
  }

  if (HasInferredTarget) {
    if (InferredTarget == CFT_Device) {
      CtorDecl->addAttr(CUDADeviceAttr::CreateImplicit(Context));
    } else if (InferredTarget == CFT_Host) {
      CtorDecl->addAttr(CUDAHostAttr::CreateImplicit(Context));
    } else {
      CtorDecl->addAttr(CUDADeviceAttr::CreateImplicit(Context));
      CtorDecl->addAttr(CUDAHostAttr::CreateImplicit(Context));
    }
  }
}
