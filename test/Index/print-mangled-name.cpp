// RUN: c-index-test -test-print-mangled-name %s | FileCheck %s

int foo(int, int);
// CHECK: mangled=_Z3fooii

int foo(float, int);
// CHECK: mangled=_Z3foofi

struct S {
  int x, y;
};

int foo(S, S&);
// CHECK: mangled=_Z3foo1SRS
