// RUN: c-index-test -test-print-mangle-itanium %s | FileCheck %s --check-prefix=ITANIUM
// RUN: c-index-test -test-print-mangle-microsoft %s | FileCheck %s --check-prefix=MICROSOFT

int foo(int, int);
// ITANIUM: mangled=_Z3fooii
// MICROSOFT: mangled={{.*}}foo@@YAHHH

int foo(float, int);
// ITANIUM: mangled=_Z3foofi
// MICROSOFT: mangled={{.*}}foo@@YAHMH

struct S {
  int x, y;
};
// ITANIUM: StructDecl{{.*}}mangled=]
// MICROSOFT: StructDecl{{.*}}mangled=]

int foo(S, S&);
// ITANIUM: mangled=_Z3foo1SRS
// MICROSOFT: mangled={{.*}}foo@@YAHUS
