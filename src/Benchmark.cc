#include <cstring>
#include <string>
#include <iostream>

#include <Benchmark.h>

extern char** environ;

void print_env()
{
  int   i          = 1;
  char* env_var_kv = *environ;

  for (; env_var_kv != 0; ++i) {
    // Split into key and value:
    char*       flag_name_cstr  = env_var_kv;
    char*       flag_value_cstr = std::strstr(env_var_kv, "=");
    int         flag_name_len   = flag_value_cstr - flag_name_cstr;
    std::string flag_name(flag_name_cstr, flag_name_cstr + flag_name_len);
    std::string flag_value(flag_value_cstr + 1);

    if (std::strstr(flag_name.c_str(), "OMPI_") ||
        std::strstr(flag_name.c_str(), "I_MPI_")) {
      std::cout << flag_name << " = " << flag_value << "\n";
    }

    env_var_kv = *(environ + i);
  }
}

bool operator<(StringDoublePair const& lhs, StringDoublePair const& rhs)
{
  return lhs.second < rhs.second;
}

std::ostream& operator<<(std::ostream& os, StringDoublePair const& p)
{
  os << "{" << p.first << ", " << p.second << "}";
  return os;
}

