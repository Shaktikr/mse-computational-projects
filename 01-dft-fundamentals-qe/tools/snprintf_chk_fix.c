/* Work-around for Ubuntu 24.04 quantum-espresso 6.7: get_md5() calls
 * snprintf(md5+2*i, sizeof(md5), ...) with sizeof(char*) as the size, which glibc's
 * _FORTIFY_SOURCE=3 flags as an overflow and aborts pw.x while reading the pseudopotential.
 * This preload makes __snprintf_chk behave like plain snprintf (the pre-fortify behaviour). */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdarg.h>
int __snprintf_chk(char *s, size_t maxlen, int flag, size_t slen, const char *fmt, ...) {
    (void)flag; (void)slen;
    va_list ap; va_start(ap, fmt);
    int r = vsnprintf(s, maxlen, fmt, ap);
    va_end(ap);
    return r;
}
