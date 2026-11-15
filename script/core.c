#include <math.h>
#include <string.h>
#define NP 11

/* p: mu l0 l1 l2 lk phi psi omega alpha gamma beta
   vtype 0: GJR-GARCH (gamma = 0 gives GARCH), vtype 1: EGARCH */

double filt(const double *r, int T, const double *p, double kappa, int vtype,
            const double *fI, const double *fIt, const double *fsg,
            double *grad, double *scores, double *s2o, double *eo, double *llo,
            double *Io, double *Ito, double s2init)
{
    double mu = p[0], l0 = p[1], l1 = p[2], l2 = p[3], lk = p[4], phi = p[5], psi = p[6];
    double om = p[7], a = p[8], g = p[9], b = p[10];
    const double Ez = 0.7978845608028654;
    double s2p = s2init, ep = r[0] - mu - (l0 + l1) * s2init;
    double dsp[NP], dep[NP], ds[NP], dm[NP], de[NP];
    int dograd = (grad || scores), clipped = 0;

    if (dograd) {
        memset(dsp, 0, sizeof dsp); memset(dep, 0, sizeof dep);
        dep[0] = -1; dep[1] = -s2init; dep[2] = -s2init;
        if (grad) memset(grad, 0, NP * sizeof(double));
        if (scores) memset(scores, 0, NP * sizeof(double));
    }
    double Ip = ep < 0 ? 1.0 : 0.0, Itp = (ep / sqrt(s2p)) < kappa ? 1.0 : 0.0;
    if (fI) Ip = fI[0];
    if (fIt) Itp = fIt[0];
    if (s2o) { s2o[0] = s2init; eo[0] = ep; llo[0] = 0; }
    if (Io) { Io[0] = Ip; Ito[0] = Itp; }

    double tot = 0;
    for (int t = 1; t < T; t++) {
        double s2, zp = ep / sqrt(s2p);
        int floored = 0;
        if (vtype == 0) {
            s2 = om + (a + g * Ip) * ep * ep + b * s2p;
        } else {
            double az = fsg ? fsg[t - 1] * zp : fabs(zp);
            double ls = om + b * log(s2p) + a * (az - Ez) + g * zp;
            clipped = 0;
            if (ls > 50) { ls = 50; clipped = 1; }
            if (ls < -50) { ls = -50; clipped = 1; }
            s2 = exp(ls);
        }
        if (s2 < 1e-8) { s2 = 1e-8; floored = 1; }

        double W = l1 + l2 * Ip + lk * Itp;
        double e = r[t] - (mu + l0 * s2 + W * s2p + phi * ep + psi * ep * ep);
        double ll = -0.5 * (1.8378770664093453 + log(s2) + e * e / s2);
        tot += ll;

        if (dograd) {
            if (vtype == 0) {
                double c = 2.0 * (a + g * Ip) * ep;
                for (int j = 0; j < NP; j++) ds[j] = floored ? 0.0 : c * dep[j] + b * dsp[j];
                if (!floored) { ds[7] += 1.0; ds[8] += ep * ep; ds[9] += Ip * ep * ep; ds[10] += s2p; }
            } else {
                double sp = sqrt(s2p), hp = log(s2p);
                double sg = fsg ? fsg[t - 1] : (zp > 0) - (zp < 0), k = a * sg + g;
                for (int j = 0; j < NP; j++) {
                    double dhp = dsp[j] / s2p, dz = dep[j] / sp - 0.5 * zp * dhp;
                    ds[j] = (floored || clipped) ? 0.0 : s2 * (b * dhp + k * dz);
                }
                if (!(floored || clipped)) {
                    ds[7] += s2; ds[8] += s2 * (sg * zp - Ez); ds[9] += s2 * zp; ds[10] += s2 * hp;
                }
            }
            double c2 = phi + 2.0 * psi * ep;
            for (int j = 0; j < NP; j++) dm[j] = l0 * ds[j] + W * dsp[j] + c2 * dep[j];
            dm[0] += 1.0; dm[1] += s2; dm[2] += s2p; dm[3] += Ip * s2p; dm[4] += Itp * s2p;
            dm[5] += ep; dm[6] += ep * ep;
            for (int j = 0; j < NP; j++) {
                de[j] = -dm[j];
                double dl = -0.5 * (ds[j] / s2 + 2.0 * e * de[j] / s2 - e * e * ds[j] / (s2 * s2));
                if (grad) grad[j] -= dl;
                if (scores) scores[t * NP + j] = dl;
            }
            memcpy(dsp, ds, sizeof ds); memcpy(dep, de, sizeof de);
        }

        double In = e < 0 ? 1.0 : 0.0, Itn = (e / sqrt(s2)) < kappa ? 1.0 : 0.0;
        if (fI) In = fI[t];
        if (fIt) Itn = fIt[t];
        if (s2o) { s2o[t] = s2; eo[t] = e; llo[t] = ll; }
        if (Io) { Io[t] = In; Ito[t] = Itn; }
        s2p = s2; ep = e; Ip = In; Itp = Itn;
    }
    return -tot;
}

void sim(const double *z, int n, const double *p, double kappa, int vtype, double s2init,
         double *r, double *s2o, double *eo)
{
    double mu = p[0], l0 = p[1], l1 = p[2], l2 = p[3], lk = p[4], phi = p[5], psi = p[6];
    double om = p[7], a = p[8], g = p[9], b = p[10];
    const double Ez = 0.7978845608028654;
    double s2p = s2init, ep = sqrt(s2init) * z[0];
    r[0] = mu + (l0 + l1) * s2init + ep; s2o[0] = s2init; eo[0] = ep;
    for (int t = 1; t < n; t++) {
        double Ip = ep < 0 ? 1.0 : 0.0, zp = ep / sqrt(s2p), Itp = zp < kappa ? 1.0 : 0.0, s2;
        if (vtype == 0) {
            s2 = om + (a + g * Ip) * ep * ep + b * s2p;
        } else {
            double ls = om + b * log(s2p) + a * (fabs(zp) - Ez) + g * zp;
            if (ls > 50) ls = 50;
            s2 = exp(ls);
        }
        if (s2 < 1e-8) s2 = 1e-8;
        double m = mu + l0 * s2 + (l1 + l2 * Ip + lk * Itp) * s2p + phi * ep + psi * ep * ep;
        double e = sqrt(s2) * z[t];
        r[t] = m + e; s2o[t] = s2; eo[t] = e; s2p = s2; ep = e;
    }
}
