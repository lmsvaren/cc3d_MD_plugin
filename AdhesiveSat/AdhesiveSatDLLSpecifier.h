

#ifndef ADHESIVESAT_EXPORT_H

#define ADHESIVESAT_EXPORT_H

    #if defined(_WIN32)

      #ifdef AdhesiveSatShared_EXPORTS

          #define ADHESIVESAT_EXPORT __declspec(dllexport)

          #define ADHESIVESAT_EXPIMP_TEMPLATE

      #else

          #define ADHESIVESAT_EXPORT __declspec(dllimport)

          #define ADHESIVESAT_EXPIMP_TEMPLATE extern

      #endif

    #else

         #define ADHESIVESAT_EXPORT

         #define ADHESIVESAT_EXPIMP_TEMPLATE

    #endif

#endif

