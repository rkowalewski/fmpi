#ifndef FMPI_DETAIL_CAPTURE_HPP
#define FMPI_DETAIL_CAPTURE_HPP

#include <tuple>

namespace fmpi {

//==============================================================================================
//                                   class Capture
//==============================================================================================
/// @class Capture
/// @brief Class allowing lambda parameter captures.
/// @note For internal use only.
template <typename RET, typename FUNC, typename... ARGS>
class Capture {
 public:
  template <typename F, typename... T>
  Capture(F&& func, T&&... args);

  template <typename... T>
  RET operator()(T&&... t);

 private:
  FUNC                _func;
  std::tuple<ARGS...> _args;
};

// Helper function
template <typename RET, typename FUNC, typename... ARGS>
Capture<RET, FUNC, ARGS...> makeCapture(FUNC&& func, ARGS&&... args);

//==============================================================================================
//                                   class Function
//==============================================================================================
/// @class Function
/// @brief Similar implementation to std::function except that it allows
/// capture of non-copyable types.
/// @note For internal use only.
template <typename SIGNATURE, size_t STORAGE_SIZE = 128>
class Function;

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
class Function<RET(ARGS...), STORAGE_SIZE> {
  using Func       = RET (*)(ARGS...);
  using Callback   = RET (*)(void*, ARGS...);
  using Destructor = void (*)(void*);

 public:
  Function() = default;
  // Ctors
  Function(RET (*ptr)(ARGS...));  // construct with function pointer
  template <typename FUNCTOR>
  Function(FUNCTOR&& functor);  // construct with functor
  Function(const Function<RET(ARGS...)>& other) = delete;
  Function(Function<RET(ARGS...)>&& other);
  Function& operator=(const Function<RET(ARGS...), STORAGE_SIZE>& other) =
      delete;
  Function& operator=(Function<RET(ARGS...), STORAGE_SIZE>&& other);
  ~Function();

  // Methods
  RET      operator()(ARGS... args);
  explicit operator bool() const;

 private:
  static void dummy(void*) {
  }

  template <typename FUNCTOR>
  void initFunctor(FUNCTOR&& functor, std::true_type);

  template <typename FUNCTOR>
  void initFunctor(FUNCTOR&& functor, std::false_type);

  union Data {
    Data() {
      // Default constructor
    }
    void*                                             ptr_;
    typename std::aligned_storage<STORAGE_SIZE>::type storage_;
  };

  Data data_;

  void*      callable_{nullptr};
  Callback   invoker_{nullptr};
  Destructor destructor_{dummy};
};

//==============================================================================================
//                                   class Capture
//==============================================================================================
template <typename RET, typename FUNC, typename... ARGS>
template <typename F, typename... T>
Capture<RET, FUNC, ARGS...>::Capture(F&& func, T&&... args)
  : _func(std::forward<F>(func))
  , _args(std::forward<T>(args)...)  // pack
{
}

template <typename RET, typename FUNC, typename... ARGS>
template <typename... T>
RET Capture<RET, FUNC, ARGS...>::operator()(T&&... t) {
  return std::apply(
      _func, std::tuple_cat(_args, std::forward_as_tuple(t...)));  // fwd
}

template <typename RET, typename FUNC, typename... ARGS>
Capture<RET, FUNC, ARGS...> makeCapture(FUNC&& func, ARGS&&... args) {
  return Capture<RET, FUNC, ARGS...>(
      std::forward<FUNC>(func), std::forward<ARGS>(args)...);
}

//==============================================================================================
//                                   class Function
//==============================================================================================

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
Function<RET(ARGS...), STORAGE_SIZE>::Function(RET (*ptr)(ARGS...))
  : callable_(reinterpret_cast<void*>(ptr)) {
  invoker_ = [](void* ptr, ARGS... args) -> RET {
    return (*reinterpret_cast<Func>(ptr))(std::forward<ARGS>(args)...);
  };
}

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
template <typename FUNCTOR>
Function<RET(ARGS...), STORAGE_SIZE>::Function(FUNCTOR&& functor) {
  initFunctor(
      std::forward<FUNCTOR>(functor), std::is_lvalue_reference<FUNCTOR>());
}

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
Function<RET(ARGS...), STORAGE_SIZE>::Function(
    Function<RET(ARGS...)>&& other) {
  *this = std::move(other);
}

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
Function<RET(ARGS...), STORAGE_SIZE>& Function<RET(ARGS...), STORAGE_SIZE>::
                                      operator=(Function<RET(ARGS...), STORAGE_SIZE>&& other) {
  if (this != &other) {
    this->~Function();  // delete current
    invoker_    = other.invoker_;
    destructor_ = other.destructor_;
    if (other.callable_ == &other.data_.storage_) {
      // copy byte-wise data
      // Note that in this case the destructor will be called on a seemingly
      // different object than the constructor however this is valid.
      data_.storage_ = other.data_.storage_;
      callable_      = &data_.storage_;
    } else {
      callable_ = other.callable_;  // steal buffer
    }
    other.callable_ = nullptr;  // disable other callable
  }
  return *this;
}

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
Function<RET(ARGS...), STORAGE_SIZE>::~Function() {
  if (destructor_) destructor_(callable_);
}

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
RET Function<RET(ARGS...), STORAGE_SIZE>::operator()(ARGS... args) {
  return invoker_(callable_, std::forward<ARGS>(args)...);
}

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
Function<RET(ARGS...), STORAGE_SIZE>::operator bool() const {
  return !!callable_;
}

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
template <typename FUNCTOR>
void Function<RET(ARGS...), STORAGE_SIZE>::initFunctor(
    FUNCTOR&& functor, std::true_type) {
  callable_ = std::addressof(functor);
  invoker_  = [](void* ptr, ARGS... args) -> RET {
    return (*reinterpret_cast<FUNCTOR*>(ptr))(std::forward<ARGS>(args)...);
  };
}

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
template <typename FUNCTOR>
void Function<RET(ARGS...), STORAGE_SIZE>::initFunctor(
    FUNCTOR&& functor, std::false_type) {
  static_assert(
      sizeof(FUNCTOR) <= STORAGE_SIZE,
      "functional object doesn't fit into internal storage");

  destructor_ = [](void* ptr) {
    if (!ptr) return;
    reinterpret_cast<FUNCTOR*>(ptr)->~FUNCTOR();  // invoke destructor
  };

  new (static_cast<void*>(&data_.storage_))
      FUNCTOR(std::forward<FUNCTOR>(functor));
  callable_ = &data_.storage_;
  invoker_  = [](void* ptr, ARGS... args) -> RET {
    return (*reinterpret_cast<FUNCTOR*>(ptr))(std::forward<ARGS>(args)...);
  };
}

}  // namespace fmpi

#endif
