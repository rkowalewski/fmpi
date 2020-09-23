#ifndef FMPI_FUNCTION_HPP
#define FMPI_FUNCTION_HPP

#include <tuple>

namespace fmpi {

template <typename F, typename... Args>
using invoke_result_t =
    std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;

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
  explicit Capture(F&& func, T&&... args);

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

  union Data;

  template <class T>
  using fits_inline = std::bool_constant<
      sizeof(T) <= sizeof(Data) && alignof(T) <= alignof(Data) &&
      std::is_nothrow_move_constructible_v<T>>;

  static_assert(sizeof(Data) <= STORAGE_SIZE);

 public:
  using return_type = RET;

  Function() = default;
  // Ctors
  explicit Function(RET (*ptr)(ARGS...));  // construct with function pointer
  template <typename FUNCTOR>
  explicit Function(FUNCTOR&& functor) noexcept(
      std::is_nothrow_move_constructible_v<FUNCTOR>);  // NOLINT
  Function(const Function<RET(ARGS...)>& other) = delete;
  Function(Function<RET(ARGS...)>&& other);  // NOLINT
  Function& operator=(const Function<RET(ARGS...), STORAGE_SIZE>& other) =
      delete;
  Function& operator=(Function<RET(ARGS...), STORAGE_SIZE>&& other) noexcept;
  ~Function();

  // Methods
  RET      operator()(ARGS... args);
  explicit operator bool() const;

  template <class F, class... CaptureArgs>
  static Function make(F&& f, CaptureArgs&&... params);

 private:
  static void dummy(void* /*unused*/) {
  }

  template <typename FUNCTOR>
  void initFunctor(FUNCTOR&& functor, std::true_type /*fits_inline*/);

  template <typename FUNCTOR>
  void initFunctor(FUNCTOR&& functor, std::false_type /*fits_inline*/);

  union Data {
    Data() {
      // Default constructor
    }
    void*                                             ptr_{};
    typename std::aligned_storage<STORAGE_SIZE>::type storage_;
  };

  Data data_;

  void*      callable_{nullptr};
  Callback   invoker_{nullptr};
  Destructor destructor_{dummy};
};

//==============================================================================================
//                                   Implementation of Capture
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
//                                   Implementation of Function
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
Function<RET(ARGS...), STORAGE_SIZE>::Function(FUNCTOR&& functor) noexcept(
    std::is_nothrow_move_constructible_v<FUNCTOR>) {
  static_assert(
      fits_inline<FUNCTOR>::value || std::is_lvalue_reference_v<FUNCTOR>);

  initFunctor(std::forward<FUNCTOR>(functor), fits_inline<FUNCTOR>{});
}

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
Function<RET(ARGS...), STORAGE_SIZE>::Function(
    Function<RET(ARGS...)>&& other) {
  *this = std::move(other);
}

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
Function<RET(ARGS...), STORAGE_SIZE>& Function<RET(ARGS...), STORAGE_SIZE>::
                                      operator=(Function<RET(ARGS...), STORAGE_SIZE>&& other) noexcept {
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
  if (destructor_ != nullptr) {
    destructor_(callable_);
  }
}

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
RET Function<RET(ARGS...), STORAGE_SIZE>::operator()(ARGS... args) {
  return invoker_(callable_, std::forward<ARGS>(args)...);
}

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
Function<RET(ARGS...), STORAGE_SIZE>::operator bool() const {
  return !(callable_ == nullptr);
}

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
template <typename FUNCTOR>
void Function<RET(ARGS...), STORAGE_SIZE>::initFunctor(
    FUNCTOR&& functor, std::false_type /*unused*/) {
  callable_ = std::addressof(functor);
  invoker_  = [](void* ptr, ARGS... args) -> RET {
    return (*reinterpret_cast<FUNCTOR*>(ptr))(std::forward<ARGS>(args)...);
  };
}

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
template <typename FUNCTOR>
void Function<RET(ARGS...), STORAGE_SIZE>::initFunctor(
    FUNCTOR&& functor, std::true_type /*unused*/) {
  destructor_ = [](void* ptr) {
    if (!ptr) {
      return;
    }
    reinterpret_cast<FUNCTOR*>(ptr)->~FUNCTOR();  // invoke destructor
  };

  new (static_cast<void*>(&data_.storage_))
      FUNCTOR(std::forward<FUNCTOR>(functor));
  callable_ = &data_.storage_;
  invoker_  = [](void* ptr, ARGS... args) -> RET {
    return (*reinterpret_cast<FUNCTOR*>(ptr))(std::forward<ARGS>(args)...);
  };
}

template <typename RET, typename... ARGS, std::size_t STORAGE_SIZE>
template <class F, class... CaptureArgs>
Function<RET(ARGS...), STORAGE_SIZE>
Function<RET(ARGS...), STORAGE_SIZE>::make(F&& f, CaptureArgs&&... params) {
  return Function<RET(ARGS...)>{makeCapture<RET>(
      std::forward<F>(f), std::forward<CaptureArgs...>(params)...)};
}

}  // namespace fmpi

#endif
