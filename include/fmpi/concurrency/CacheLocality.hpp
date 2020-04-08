#ifndef FMPI_CONCURRENCY_CACHELOCALITY_HPP
#define FMPI_CONCURRENCY_CACHELOCALITY_HPP

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace folly {
// This file contains several classes that might be useful if you are
// trying to dynamically optimize cache locality: CacheLocality reads
// cache sharing information from sysfs to determine how CPUs should be
// grouped to minimize contention, Getcpu provides fast access to the
// current CPU via __vdso_getcpu, and AccessSpreader uses these two to
// optimally spread accesses among a predetermined number of stripes.
//
// AccessSpreader<>::current(n) microbenchmarks at 22 nanos, which is
// substantially less than the cost of a cache miss.  This means that we
// can effectively use it to reduce cache line ping-pong on striped data
// structures such as IndexedMemPool or statistics counters.
//
// Because CacheLocality looks at all of the cache levels, it can be
// used for different levels of optimization.  AccessSpreader(2) does
// per-chip spreading on a dual socket system.  AccessSpreader(numCpus)
// does perfect per-cpu spreading.  AccessSpreader(numCpus / 2) does
// perfect L1 spreading in a system with hyperthreading enabled.

struct CacheLocality {
  /// 1 more than the maximum value that can be returned from sched_getcpu
  /// or getcpu.  This is the number of hardware thread contexts provided
  /// by the processors
  std::size_t numCpus{};

  /// Holds the number of caches present at each cache level (0 is
  /// the closest to the cpu).  This is the number of AccessSpreader
  /// stripes needed to avoid cross-cache communication at the specified
  /// layer.  numCachesByLevel.front() is the number of L1 caches and
  /// numCachesByLevel.back() is the number of last-level caches.
  std::vector<std::size_t> numCachesByLevel;

  /// A map from cpu (from sched_getcpu or getcpu) to an index in the
  /// range 0..numCpus-1, where neighboring locality indices are more
  /// likely to share caches then indices far away.  All of the members
  /// of a particular cache level be contiguous in their locality index.
  /// For example, if numCpus is 32 and numCachesByLevel.back() is 2,
  /// then cpus with a locality index < 16 will share one last-level
  /// cache and cpus with a locality index >= 16 will share the other.
  std::vector<std::size_t> localityIndexByCpu;

  /// Returns the best CacheLocality information available for the current
  /// system, cached for fast access.  This will be loaded from sysfs if
  /// possible, otherwise it will be correct in the number of CPUs but
  /// not in their sharing structure.
  ///
  /// If you are into yo dawgs, this is a shared cache of the local
  /// locality of the shared caches.
  ///
  /// The template parameter here is used to allow injection of a
  /// repeatable CacheLocality structure during testing.  Rather than
  /// inject the type of the CacheLocality provider into every data type
  /// that transitively uses it, all components select between the default
  /// sysfs implementation and a deterministic implementation by keying
  /// off the type of the underlying atomic.  See DeterministicScheduler.
  template <template <typename> class Atom = std::atomic>
  static const CacheLocality& system();

  /// Reads CacheLocality information from a tree structured like
  /// the sysfs filesystem.  The provided function will be evaluated
  /// for each sysfs file that needs to be queried.  The function
  /// should return a string containing the first line of the file
  /// (not including the newline), or an empty string if the file does
  /// not exist.  The function will be called with paths of the form
  /// /sys/devices/system/cpu/cpu*/cache/index*/{type,shared_cpu_list} .
  /// Throws an exception if no caches can be parsed at all.
  static CacheLocality readFromSysfsTree(
      const std::function<std::string(std::string)>& mapping);

  /// Reads CacheLocality information from the real sysfs filesystem.
  /// Throws an exception if no cache information can be loaded.
  static CacheLocality readFromSysfs();

  /// readFromProcCpuinfo(), except input is taken from memory rather
  /// than the file system.
  static CacheLocality readFromProcCpuinfoLines(
      std::vector<std::string> const& lines);

  /// Returns an estimate of the CacheLocality information by reading
  /// /proc/cpuinfo.  This isn't as accurate as readFromSysfs(), but
  /// is a lot faster because the info isn't scattered across
  /// hundreds of files.  Throws an exception if no cache information
  /// can be loaded.
  static CacheLocality readFromProcCpuinfo();

  /// Returns a usable (but probably not reflective of reality)
  /// CacheLocality structure with the specified number of cpus and a
  /// single cache level that associates one cpu per cache.
  static CacheLocality uniform(std::size_t numCpus);
};

#if 0

/// Knows how to derive a function pointer to the VDSO implementation of
/// getcpu(2), if available
struct Getcpu {
  /// Function pointer to a function with the same signature as getcpu(2).
  typedef int (*Func)(unsigned* cpu, unsigned* node, void* unused);

  /// Returns a pointer to the VDSO implementation of getcpu(2), if
  /// available, or nullptr otherwise.  This function may be quite
  /// expensive, be sure to cache the result.
  static Func resolveVdsoFunc();
};

/// Returns an estimate of the CacheLocality information by reading
/// /proc/cpuinfo.  This isn't as accurate as readFromSysfs(), but
/// is a lot faster because the info isn't scattered across
/// hundreds of files.  Throws an exception if no cache information
/// can be loaded.
static CacheLocality readFromProcCpuinfo();

/// Returns a usable (but probably not reflective of reality)
/// CacheLocality structure with the specified number of cpus and a
/// single cache level that associates one cpu per cache.
static CacheLocality uniform(size_t numCpus);
#endif
}  // namespace folly
#endif
