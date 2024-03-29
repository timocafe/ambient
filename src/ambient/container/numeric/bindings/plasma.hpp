/*
 * Copyright Institute for Theoretical Physics, ETH Zurich 2015.
 * Distributed under the Boost Software License, Version 1.0.
 *
 * Permission is hereby granted, free of charge, to any person or organization
 * obtaining a copy of the software and accompanying documentation covered by
 * this license (the "Software") to use, reproduce, display, distribute,
 * execute, and transmit the Software, and to prepare derivative works of the
 * Software, and to permit third-parties to whom the Software is furnished to
 * do so, all subject to the following:
 *
 * The copyright notices in the Software and this entire statement, including
 * the above license grant, this restriction and the following disclaimer,
 * must be included in all copies of the Software, in whole or in part, and
 * all derivative works of the Software, unless such copies or derivative
 * works are solely in the form of machine-executable object code generated by
 * a source language processor.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef AMBIENT_CONTAINER_NUMERIC_BINDINGS_PLASMA
#define AMBIENT_CONTAINER_NUMERIC_BINDINGS_PLASMA

#define PLASMA_IB        64
#define PlasmaNoTrans    111
#define PlasmaTrans      112
#define PlasmaConjTrans  113
#define PlasmaUpper      121
#define PlasmaLower      122
#define PlasmaUpperLower 123
#define PlasmaLeft       141
#define PlasmaRight      142
#define PlasmaForward    391
#define PlasmaColumnwise 401

typedef int PLASMA_enum;

template <PLASMA_enum TR>
struct trans_type {
    const static int value = TR;
};

template <PLASMA_enum UL>
struct ul_type {
    const static int value = UL;
};

template <class T>
struct trans_helper {
    const static int PlasmaTransValue = PlasmaTrans;
};

template <class T>
struct trans_helper<std::complex<T> > {
    const static int PlasmaTransValue = PlasmaConjTrans;
};

extern "C" {
    int    CORE_dgeqrt(int M, int N, int IB,
        double* A, int LDA,
        double* T, int LDT,
        double* TAU, double* WORK);
    int    CORE_zgeqrt(int M, int N, int IB,
        std::complex<double>* A, int LDA,
        std::complex<double>* T, int LDT,
        std::complex<double>* TAU, std::complex<double>* WORK);
    int    CORE_dormqr(PLASMA_enum side, PLASMA_enum trans,
        int M, int N, int K, int IB,
        const double* V, int LDV,
        const double* T, int LDT,
        double* C, int LDC,
        double* WORK, int LDWORK);
    int    CORE_zunmqr(PLASMA_enum side, PLASMA_enum trans,
        int M, int N, int K, int IB,
        const std::complex<double>* V, int LDV,
        const std::complex<double>* T, int LDT,
        std::complex<double>* C, int LDC,
        std::complex<double>* WORK, int LDWORK);
    int    CORE_dtsqrt(int M, int N, int IB,
        double* A1, int LDA1,
        double* A2, int LDA2,
        double* T, int LDT,
        double* TAU, double* WORK);
    int    CORE_ztsqrt(int M, int N, int IB,
        std::complex<double>* A1, int LDA1,
        std::complex<double>* A2, int LDA2,
        std::complex<double>* T, int LDT,
        std::complex<double>* TAU, std::complex<double>* WORK);
    int    CORE_dtsmqr(PLASMA_enum side, PLASMA_enum trans,
        int M1, int N1, int M2, int N2, int K, int IB,
        double* A1, int LDA1,
        double* A2, int LDA2,
        const double* V, int LDV,
        const double* T, int LDT,
        double* WORK, int LDWORK);
    int    CORE_ztsmqr(PLASMA_enum side, PLASMA_enum trans,
        int M1, int N1, int M2, int N2, int K, int IB,
        std::complex<double>* A1, int LDA1,
        std::complex<double>* A2, int LDA2,
        const std::complex<double>* V, int LDV,
        const std::complex<double>* T, int LDT,
        std::complex<double>* WORK, int LDWORK);
    int    CORE_dgelqt(int M, int N, int IB,
        double* A, int LDA,
        double* T, int LDT,
        double* TAU,
        double* WORK);
    int    CORE_zgelqt(int M, int N, int IB,
        std::complex<double>* A, int LDA,
        std::complex<double>* T, int LDT,
        std::complex<double>* TAU,
        std::complex<double>* WORK);
    int    CORE_dormlq(PLASMA_enum side, PLASMA_enum trans,
        int M, int N, int IB, int K,
        const double* V, int LDV,
        const double* T, int LDT,
        double* C, int LDC,
        double* WORK, int LDWORK);
    int    CORE_zunmlq(PLASMA_enum side, PLASMA_enum trans,
        int M, int N, int IB, int K,
        const std::complex<double>* V, int LDV,
        const std::complex<double>* T, int LDT,
        std::complex<double>* C, int LDC,
        std::complex<double>* WORK, int LDWORK);
    int    CORE_dtslqt(int M, int N, int IB,
        double* A1, int LDA1,
        double* A2, int LDA2,
        double* T, int LDT,
        double* TAU, double* WORK);
    int    CORE_ztslqt(int M, int N, int IB,
        std::complex<double>* A1, int LDA1,
        std::complex<double>* A2, int LDA2,
        std::complex<double>* T, int LDT,
        std::complex<double>* TAU, std::complex<double>* WORK);
    int    CORE_dtsmlq(PLASMA_enum side, PLASMA_enum trans,
        int M1, int N1, int M2, int N2, int K, int IB,
        double* A1, int LDA1,
        double* A2, int LDA2,
        const double* V, int LDV,
        const double* T, int LDT,
        double* WORK, int LDWORK);
    int    CORE_ztsmlq(PLASMA_enum side, PLASMA_enum trans,
        int M1, int N1, int M2, int N2, int K, int IB,
        std::complex<double>* A1, int LDA1,
        std::complex<double>* A2, int LDA2,
        const std::complex<double>* V, int LDV,
        const std::complex<double>* T, int LDT,
        std::complex<double>* WORK, int LDWORK);
    void  CORE_dlaset2(PLASMA_enum uplo, int n1, int n2, double alpha,
        double* tileA, int ldtilea);
    void  CORE_zlaset2(PLASMA_enum uplo, int n1, int n2, std::complex<double> alpha,
        std::complex<double>* tileA, int ldtilea);
}

namespace ambient {
    inline namespace numeric {
        namespace plasma {

            template<class T>
            struct lapack;

            template<>
            struct lapack<double> {
                typedef double T;
                static void geqrt(int m, int n, int ib, T* a, int lda, T* t, int ldt, T* tau, T* work) {
                    CORE_dgeqrt(m, n, ib, a, lda, t, ldt, tau, work);
                }

                static void ormqr(int side, int trans, int m, int n, int k, int in, const T* a, int lda, const T* t, int ldt, T* c, int ldc, T* work, int ldwork) {
                    CORE_dormqr(side, trans, m, n, k, in, a, lda, t, ldt, c, ldc, work, ldwork);
                }

                static void tsqrt(int m, int n, int ib, T* a1, int lda1, T* a2, int lda2, T* t, int ldt, T* tau, T* work) {
                    CORE_dtsqrt(m, n, ib, a1, lda1, a2, lda2, t, ldt, tau, work);
                }

                static void tsmqr(int side, int trans, int m1, int n1, int m2, int n2, int k, int ib, T* a1, int lda1, T* a2, int lda2, T* v, int ldv, T* t, int ldt, T* work, int ldwork) {
                    CORE_dtsmqr(side, trans, m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, v, ldv, t, ldt, work, ldwork);
                }

                static void gelqt(int m, int n, int ib, T* a, int lda, T* t, int ldt, T* tau, T* work) {
                    CORE_dgelqt(m, n, ib, a, lda, t, ldt, tau, work);
                }

                static void ormlq(int side, int trans, int m, int n, int k, int in, const T* a, int lda, const T* t, int ldt, T* c, int ldc, T* work, int ldwork) {
                    CORE_dormlq(side, trans, m, n, k, in, a, lda, t, ldt, c, ldc, work, ldwork);
                }

                static void tslqt(int m, int n, int ib, T* a1, int lda1, T* a2, int lda2, T* t, int ldt, T* tau, T* work) {
                    CORE_dtslqt(m, n, ib, a1, lda1, a2, lda2, t, ldt, tau, work);
                }

                static void tsmlq(int side, int trans, int m1, int n1, int m2, int n2, int k, int ib, T* a1, int lda1, T* a2, int lda2, T* v, int ldv, T* t, int ldt, T* work, int ldwork) {
                    CORE_dtsmlq(side, trans, m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, v, ldv, t, ldt, work, ldwork);
                }

                static void laset2(int uplo, int n1, int n2, T alfa, T* A, int lda) {
                    CORE_dlaset2(uplo, n1, n2, alfa, A, lda);
                }

            };

            template<>
            struct lapack<std::complex<double> > {
                typedef std::complex<double> T;
                static void geqrt(int m, int n, int ib, T* a, int lda, T* t, int ldt, T* tau, T* work) {
                    CORE_zgeqrt(m, n, ib, a, lda, t, ldt, tau, work);
                }

                static void ormqr(int side, int trans, int m, int n, int k, int in, const T* a, int lda, const T* t, int ldt, T* c, int ldc, T* work, int ldwork) {
                    CORE_zunmqr(side, trans, m, n, k, in, a, lda, t, ldt, c, ldc, work, ldwork);
                }

                static void tsqrt(int m, int n, int ib, T* a1, int lda1, T* a2, int lda2, T* t, int ldt, T* tau, T* work) {
                    CORE_ztsqrt(m, n, ib, a1, lda1, a2, lda2, t, ldt, tau, work);
                }

                static void tsmqr(int side, int trans, int m1, int n1, int m2, int n2, int k, int ib, T* a1, int lda1, T* a2, int lda2, T* v, int ldv, T* t, int ldt, T* work, int ldwork) {
                    CORE_ztsmqr(side, trans, m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, v, ldv, t, ldt, work, ldwork);
                }

                static void gelqt(int m, int n, int ib, T* a, int lda, T* t, int ldt, T* tau, T* work) {
                    CORE_zgelqt(m, n, ib, a, lda, t, ldt, tau, work);
                }

                static void ormlq(int side, int trans, int m, int n, int k, int in, const T* a, int lda, const T* t, int ldt, T* c, int ldc, T* work, int ldwork) {
                    CORE_zunmlq(side, trans, m, n, k, in, a, lda, t, ldt, c, ldc, work, ldwork);
                }

                static void tslqt(int m, int n, int ib, T* a1, int lda1, T* a2, int lda2, T* t, int ldt, T* tau, T* work) {
                    CORE_ztslqt(m, n, ib, a1, lda1, a2, lda2, t, ldt, tau, work);
                }

                static void tsmlq(int side, int trans, int m1, int n1, int m2, int n2, int k, int ib, T* a1, int lda1, T* a2, int lda2, T* v, int ldv, T* t, int ldt, T* work, int ldwork) {
                    CORE_ztsmlq(side, trans, m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, v, ldv, t, ldt, work, ldwork);
                }

                static void laset2(int uplo, int n1, int n2, T alfa, T* A, int lda) {
                    CORE_zlaset2(uplo, n1, n2, alfa, A, lda);
                }
            };

        }
    }
}

#endif
