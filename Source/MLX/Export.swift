// Copyright © 2025 Apple Inc.

import Cmlx
import Foundation

/// Export a function to a `.mlxfn` file.
///
/// For example this defines a function, writes it to a file, imports
/// it and evaluates it.
///
/// ```swift
/// func f(arrays: [MLXArray]) -> [MLXArray] {
///     [arrays[0] * arrays[1]]
/// }
///
/// let x = MLXArray(1)
/// let y = MLXArray([1, 2, 3])
///
/// try exportFunction(to: url, f)(x, y: y)
///
/// // load it back in
/// let f2 = try importFunction(from: url)
///
/// let a = MLXArray(10)
/// let b = MLXArray([5, 10, 20])
///
/// // call it -- the shapes and labels have to match
/// let r = try f2(a, y: b)[0]
/// ```
///
/// - Parameters:
///   - url: file url to write the `.mlxfn` file
///   - shapeless: if `true` the function allows inputs with variable shapes
///   - f: the function to capture
/// - Returns: a helper (``FunctionExporterSingle``) that records the call
///
/// ### See Also
/// - ``exportFunctions(to:shapeless:_:build:)``
/// - ``importFunction(from:)``
public func exportFunction(
    to url: URL, shapeless: Bool = false, _ f: @escaping ([MLXArray]) -> [MLXArray]
) -> FunctionExporterSingle {
    FunctionExporterSingle(url: url, shapeless: shapeless, f: f)
}

/// Export multiple traces of a function to a `.mlxfn` file.
///
/// For example this defines a function, writes it to a file, imports
/// it and evaluates it.
///
/// ```swift
/// func f(_ arrays: [MLXArray]) -> [MLXArray] {
///     [arrays.dropFirst().reduce(arrays[0], +)]
/// }
///
/// let x = MLXArray(1)
///
/// try exportFunctions(to: url, shapeless: true, f) { export in
///     try export(x)
///     try export(x, x)
///     try export(x, x, x)
/// }
///
/// // load it back in
/// let f2 = try importFunction(from: url)
///
/// let a = MLXArray([10, 10, 10])
/// let b = MLXArray([5, 10, 20])
/// let c = MLXArray([1, 2, 3])
///
/// // r1 = a
/// let r1 = try f2(a)[0]
///
/// // r2 = a + b
/// let r2 = try f2(a, b)[0]
///
/// // r3 = a + b + c
/// let r3 = try f2(a, b, c)[0]
/// ```
///
/// - Parameters:
///   - url: file url to write the `.mlxfn` file
///   - shapeless: if `true` the function allows inputs with variable shapes
///   - f: the function to capture
///   - build: closure for recording the calls
///
/// ### See Also
/// - ``exportFunction(to:shapeless:_:)``
/// - ``importFunction(from:)``
public func exportFunctions(
    to url: URL, shapeless: Bool = false, _ f: @escaping ([MLXArray]) -> [MLXArray],
    build: (FunctionExporterMultiple) throws -> Void
) throws {
    let exporter = try FunctionExporterMultiple(url: url, shapeless: shapeless, f: f)
    try build(exporter)
}

/// A helper for ``exportFunction(to:shapeless:_:)``.
///
/// This records the call to the function and saves it to the file.
///
/// ```swift
/// func f(arrays: [MLXArray]) -> [MLXArray] {
///     [arrays[0] * arrays[1]]
/// }
///
/// let x = MLXArray(1)
/// let y = MLXArray([1, 2, 3])
///
/// // the (x, y: y) is calling this object
/// try exportFunction(to: url, f)(x, y: y)
/// ```
///
/// ### See Also
/// - ``exportFunction(to:shapeless:_:)``
@dynamicCallable
public final class FunctionExporterSingle {
    let url: URL
    let shapeless: Bool
    let f: ([MLXArray]) -> [MLXArray]

    internal init(url: URL, shapeless: Bool, f: @escaping ([MLXArray]) -> [MLXArray]) {
        self.url = url
        self.shapeless = shapeless
        self.f = f
    }

    public func dynamicallyCall(withKeywordArguments args: KeyValuePairs<String, MLXArray>) throws {
        let positionalArgs = mlx_vector_array_new()
        defer { mlx_vector_array_free(positionalArgs) }
        for (key, value) in args {
            if key.isEmpty {
                mlx_vector_array_append_value(positionalArgs, value.ctx)
            }
        }

        let keys = args.compactMap { $0.key.isEmpty ? nil : $0.key }
        let kwargs = new_mlx_array_map(
            Dictionary(
                args.compactMap { $0.key.isEmpty ? nil : ($0.key, $0.value) },
                uniquingKeysWith: { a, b in a }))
        defer { mlx_map_string_to_array_free(kwargs) }

        let closure = new_mlx_kwargs_closure(keys: keys, f)
        defer { mlx_closure_kwargs_free(closure) }

        _ = try withError {
            mlx_export_function_kwargs(url.path, closure, positionalArgs, kwargs, shapeless)
        }
    }
}

/// A helper for ``exportFunctions(to:shapeless:_:build:)``.
///
/// This records the call to the function and saves it to the file.
///
/// ```swift
/// func f(_ arrays: [MLXArray]) -> [MLXArray] {
///     [arrays.dropFirst().reduce(arrays[0], +)]
/// }
///
/// let x = MLXArray(1)
///
/// // the export parameter is a FunctionExporterMultiple
/// try exportFunctions(to: url, shapeless: true, f) { export in
///     try export(x)
///     try export(x, x)
///     try export(x, x, x)
/// }
/// ```
///
/// ### See Also
/// - ``exportFunctions(to:shapeless:_:build:)``
@dynamicCallable
public final class FunctionExporterMultiple {
    let exporter: mlx_function_exporter

    internal init(url: URL, shapeless: Bool = false, f: @escaping ([MLXArray]) -> [MLXArray]) throws
    {
        let closure = new_mlx_closure(f)
        defer { mlx_closure_free(closure) }

        self.exporter = try withError {
            mlx_function_exporter_new(url.path, closure, shapeless)
        }
    }

    deinit {
        mlx_function_exporter_free(exporter)
    }

    public func dynamicallyCall(withKeywordArguments args: KeyValuePairs<String, MLXArray>) throws {
        let positionalArgs = mlx_vector_array_new()
        defer { mlx_vector_array_free(positionalArgs) }
        for (key, value) in args {
            if key.isEmpty {
                mlx_vector_array_append_value(positionalArgs, value.ctx)
            }
        }

        let kwargs = new_mlx_array_map(
            Dictionary(
                args.compactMap { $0.key.isEmpty ? nil : ($0.key, $0.value) },
                uniquingKeysWith: { a, b in a }))
        defer { mlx_map_string_to_array_free(kwargs) }

        _ = try withError {
            mlx_function_exporter_apply_kwargs(exporter, positionalArgs, kwargs)
        }
    }
}

// MARK: - Callback-mode export

/// One event in the MLX export stream.
///
/// Projection of `mlx::core::ExportCallbackInput` (see
/// `mlx/export.h:34-42`). One ``MLXExportEvent`` is emitted per entry in the
/// underlying `unordered_map` for a given callback invocation — a typical
/// primitive event surfaces as `.primitive(name:inputs:outputs:arguments:)`
/// after the callback collects the related map entries.
///
/// Adding or removing a case is a breaking change — the enum mirrors the C++
/// variant arms 1:1 so the parser layer can dispatch exhaustively.
///
/// ### See Also
/// - ``exportFunction(callback:shapeless:_:)``
///
/// > Note: Not `Sendable` — ``namedArrays(_:)`` carries `MLXArray`, which
/// > is a reference type. Stay on the thread that invoked the callback.
public enum MLXExportEvent {
    /// `vector<tuple<string, Shape, Dtype>>` — `inputs` / `outputs` payloads.
    case tensorSpecs([MLXExportTensorSpec])

    /// `vector<pair<string, array>>` — `constants` payload.
    case namedArrays([(name: String, array: MLXArray)])

    /// `vector<pair<string, string>>` — `keyword_inputs` payload.
    case namedStrings([(first: String, second: String)])

    /// `vector<StateT>` — `arguments` payload of a `primitive` record.
    case primitiveStates([MLXPrimitiveStateValue])

    /// `string` — `name` and `type` payloads.
    case string(String)
}

/// One tensor spec entry inside ``MLXExportEvent/tensorSpecs(_:)``.
public struct MLXExportTensorSpec: Sendable {
    public let name: String
    public let shape: [Int32]
    public let dtype: DType

    public init(name: String, shape: [Int32], dtype: DType) {
        self.name = name
        self.shape = shape
        self.dtype = dtype
    }
}

/// One ``MLXExportEvent/primitiveStates(_:)`` entry.
///
/// Projection of `mlx::core::StateT` (see `mlx/export.h:18-32`). Every
/// variant arm has a corresponding case so the consumer can switch
/// exhaustively.
public enum MLXPrimitiveStateValue: Sendable {
    case bool(Bool)
    case int(Int32)
    case size(Int)
    case float(Float)
    case double(Double)
    case dtype(DType)
    case shape([Int32])
    case strides([Int64])
    case intArray([Int32])
    case sizeArray([Int])
    case tripleBoolArray([(Bool, Bool, Bool)])
    case mixedScalarArray([MLXPrimitiveMixedScalar])
    case optionalFloat(Float?)
    case string(String)
}

/// `variant<bool, int, float>` arm inside
/// ``MLXPrimitiveStateValue/mixedScalarArray(_:)``.
public enum MLXPrimitiveMixedScalar: Sendable {
    case bool(Bool)
    case int(Int32)
    case float(Float)
}

/// One callback invocation's payload — a list of `(key, event)` pairs in
/// the order yielded by the underlying `std::unordered_map`.
///
/// Each invocation describes a single record in the export stream. The
/// `"type"` entry tags the record kind; the remaining entries carry the
/// payload. The per-kind key layout (from `mlx/export.cpp`) is:
///
/// | `type` value         | data keys                                                     | event case(s)                       |
/// |----------------------|---------------------------------------------------------------|-------------------------------------|
/// | `"inputs"`           | `"inputs"`                                                    | `.tensorSpecs`                      |
/// | `"keyword_inputs"`   | `"keywords"` (**not** `"keyword_inputs"`)                     | `.namedStrings`                     |
/// | `"outputs"`          | `"outputs"`                                                   | `.tensorSpecs`                      |
/// | `"constants"`        | `"constants"`                                                 | `.namedArrays`                      |
/// | `"primitive"`        | `"name"`, `"inputs"`, `"outputs"`, `"arguments"`              | `.string`, `.tensorSpecs`, `.primitiveStates` |
///
/// In particular: for the keyword-inputs record, `"keyword_inputs"` is the
/// *value* of the `"type"` key, while the actual `(keyword, tensorName)`
/// pairs live under the `"keywords"` key. Downstream tools (e.g.
/// mlx2coreml) match this distinction.
public struct MLXExportCallbackPayload {
    public let entries: [(key: String, event: MLXExportEvent)]

    public subscript(key: String) -> MLXExportEvent? {
        entries.first { $0.key == key }?.event
    }
}

/// Capture an MLX function as a stream of primitive events.
///
/// Mirrors the callback overload of `mx.export_function` in the Python
/// bindings, which downstream graph-translation tools (e.g. mlx2coreml) use to
/// observe primitive-level structure without writing a `.mlxfn` file. Each
/// callback invocation receives one ``MLXExportCallbackPayload`` describing a
/// single record in the export stream — input spec, keyword-input map, output
/// spec, constants, or one primitive.
///
/// ```swift
/// var events: [MLXExportCallbackPayload] = []
/// try exportFunction(callback: { events.append($0) }) { arrays in
///     [arrays[0] + arrays[1]]
/// }(x, y: y)
/// ```
///
/// - Parameters:
///   - callback: invoked once per record in the export stream
///   - shapeless: if `true` the function allows inputs with variable shapes
///   - f: the function to trace
/// - Returns: a helper that records the call (mirrors
///   ``FunctionExporterSingle``)
///
/// ### See Also
/// - ``exportFunction(to:shapeless:_:)``
public func exportFunction(
    callback: @escaping (MLXExportCallbackPayload) -> Void,
    shapeless: Bool = false,
    _ f: @escaping ([MLXArray]) -> [MLXArray]
) -> FunctionCallbackExporter {
    FunctionCallbackExporter(callback: callback, shapeless: shapeless, f: f)
}

/// Helper for ``exportFunction(callback:shapeless:_:)``.
///
/// Mirrors ``FunctionExporterSingle`` — invoke with positional and keyword
/// `MLXArray` arguments via `@dynamicCallable`:
///
/// ```swift
/// try exportFunction(callback: { events.append($0) }, f)(x, y: y)
/// ```
@dynamicCallable
public final class FunctionCallbackExporter {
    let callback: (MLXExportCallbackPayload) -> Void
    let shapeless: Bool
    let f: ([MLXArray]) -> [MLXArray]

    internal init(
        callback: @escaping (MLXExportCallbackPayload) -> Void,
        shapeless: Bool,
        f: @escaping ([MLXArray]) -> [MLXArray]
    ) {
        self.callback = callback
        self.shapeless = shapeless
        self.f = f
    }

    public func dynamicallyCall(
        withKeywordArguments args: KeyValuePairs<String, MLXArray>
    ) throws {
        let positionalArgs = mlx_vector_array_new()
        defer { mlx_vector_array_free(positionalArgs) }
        for (key, value) in args where key.isEmpty {
            mlx_vector_array_append_value(positionalArgs, value.ctx)
        }

        let keys = args.compactMap { $0.key.isEmpty ? nil : $0.key }
        let kwargs = new_mlx_array_map(
            Dictionary(
                args.compactMap { $0.key.isEmpty ? nil : ($0.key, $0.value) },
                uniquingKeysWith: { a, _ in a }))
        defer { mlx_map_string_to_array_free(kwargs) }

        let funClosure = new_mlx_kwargs_closure(keys: keys, f)
        defer { mlx_closure_kwargs_free(funClosure) }

        let cbClosure = new_mlx_export_callback_closure(callback)
        defer { mlx_closure_export_callback_free(cbClosure) }

        _ = try withError {
            mlx_export_function_callback(
                cbClosure, funClosure, positionalArgs, kwargs, shapeless)
        }
    }
}

// MARK: - Callback decoders (internal)

private final class CallbackCaptureState {
    let callback: (MLXExportCallbackPayload) -> Void
    init(_ callback: @escaping (MLXExportCallbackPayload) -> Void) {
        self.callback = callback
    }
}

private func decodeCallbackInput(_ input: mlx_export_callback_input)
    -> MLXExportCallbackPayload
{
    let count = mlx_export_callback_input_size(input)
    var entries: [(key: String, event: MLXExportEvent)] = []
    entries.reserveCapacity(count)
    for index in 0 ..< count {
        guard let keyPtr = mlx_export_callback_input_key(input, index) else {
            continue
        }
        let key = String(cString: keyPtr)
        let valueHandle = mlx_export_callback_input_value(input, index)
        entries.append((key, decodeValue(valueHandle)))
    }
    return MLXExportCallbackPayload(entries: entries)
}

private func decodeValue(_ value: mlx_export_callback_value) -> MLXExportEvent {
    switch mlx_export_callback_value_get_kind(value) {
    case MLX_EXPORT_VALUE_TENSOR_SPECS:
        let count = mlx_export_callback_value_tensor_specs_size(value)
        var specs: [MLXExportTensorSpec] = []
        specs.reserveCapacity(count)
        for index in 0 ..< count {
            let name = mlx_export_callback_value_tensor_specs_name(value, index)
                .map { String(cString: $0) } ?? ""
            var shapeData: UnsafePointer<Int32>?
            var shapeSize: Int = 0
            _ = mlx_export_callback_value_tensor_specs_shape(
                &shapeData, &shapeSize, value, index)
            let shape = shapeData.map { ptr in
                Array(UnsafeBufferPointer(start: ptr, count: shapeSize))
            } ?? []
            var dtype: mlx_dtype = MLX_FLOAT32
            _ = mlx_export_callback_value_tensor_specs_dtype(&dtype, value, index)
            specs.append(MLXExportTensorSpec(
                name: name, shape: shape, dtype: DType(dtype)))
        }
        return .tensorSpecs(specs)

    case MLX_EXPORT_VALUE_NAMED_ARRAYS:
        let count = mlx_export_callback_value_named_arrays_size(value)
        var pairs: [(name: String, array: MLXArray)] = []
        pairs.reserveCapacity(count)
        for index in 0 ..< count {
            let name = mlx_export_callback_value_named_arrays_name(value, index)
                .map { String(cString: $0) } ?? ""
            var arrayCtx = mlx_array_new()
            _ = mlx_export_callback_value_named_arrays_get(&arrayCtx, value, index)
            pairs.append((name, MLXArray(arrayCtx)))
        }
        return .namedArrays(pairs)

    case MLX_EXPORT_VALUE_NAMED_STRINGS:
        let count = mlx_export_callback_value_named_strings_size(value)
        var pairs: [(first: String, second: String)] = []
        pairs.reserveCapacity(count)
        for index in 0 ..< count {
            let first = mlx_export_callback_value_named_strings_first(value, index)
                .map { String(cString: $0) } ?? ""
            let second = mlx_export_callback_value_named_strings_second(value, index)
                .map { String(cString: $0) } ?? ""
            pairs.append((first, second))
        }
        return .namedStrings(pairs)

    case MLX_EXPORT_VALUE_PRIMITIVE_STATES:
        let count = mlx_export_callback_value_states_size(value)
        var states: [MLXPrimitiveStateValue] = []
        states.reserveCapacity(count)
        for index in 0 ..< count {
            let handle = mlx_export_callback_value_states_at(value, index)
            states.append(decodeState(handle))
        }
        return .primitiveStates(states)

    case MLX_EXPORT_VALUE_STRING:
        let cstr = mlx_export_callback_value_string(value)
        return .string(cstr.map { String(cString: $0) } ?? "")

    default:
        fatalError("Unknown mlx_export_callback_value_kind — mlx version mismatch?")
    }
}

private func decodeState(_ state: mlx_export_state) -> MLXPrimitiveStateValue {
    switch mlx_export_state_get_kind(state) {
    case MLX_EXPORT_STATE_BOOL:
        var v: Bool = false
        _ = mlx_export_state_bool(&v, state)
        return .bool(v)
    case MLX_EXPORT_STATE_INT:
        var v: Int32 = 0
        _ = mlx_export_state_int(&v, state)
        return .int(v)
    case MLX_EXPORT_STATE_SIZE:
        var v: Int = 0
        _ = mlx_export_state_size(&v, state)
        return .size(v)
    case MLX_EXPORT_STATE_FLOAT:
        var v: Float = 0
        _ = mlx_export_state_float(&v, state)
        return .float(v)
    case MLX_EXPORT_STATE_DOUBLE:
        var v: Double = 0
        _ = mlx_export_state_double(&v, state)
        return .double(v)
    case MLX_EXPORT_STATE_DTYPE:
        var v: mlx_dtype = MLX_FLOAT32
        _ = mlx_export_state_dtype(&v, state)
        return .dtype(DType(v))
    case MLX_EXPORT_STATE_SHAPE:
        return .shape(decodeStateVector(state, mlx_export_state_shape))
    case MLX_EXPORT_STATE_STRIDES:
        return .strides(decodeStateVector(state, mlx_export_state_strides))
    case MLX_EXPORT_STATE_INT_ARRAY:
        return .intArray(decodeStateVector(state, mlx_export_state_int_array))
    case MLX_EXPORT_STATE_SIZE_ARRAY:
        return .sizeArray(decodeStateVector(state, mlx_export_state_size_array))
    case MLX_EXPORT_STATE_TRIPLE_BOOL_ARRAY:
        var size: Int = 0
        _ = mlx_export_state_triple_bool_array_size(&size, state)
        var out: [(Bool, Bool, Bool)] = []
        out.reserveCapacity(size)
        for index in 0 ..< size {
            var a = false, b = false, c = false
            _ = mlx_export_state_triple_bool_array_at(&a, &b, &c, state, index)
            out.append((a, b, c))
        }
        return .tripleBoolArray(out)
    case MLX_EXPORT_STATE_MIXED_SCALAR_ARRAY:
        var size: Int = 0
        _ = mlx_export_state_mixed_scalar_array_size(&size, state)
        var out: [MLXPrimitiveMixedScalar] = []
        out.reserveCapacity(size)
        for index in 0 ..< size {
            var kind: mlx_export_mixed_scalar_kind = MLX_EXPORT_MIXED_BOOL
            _ = mlx_export_state_mixed_scalar_array_kind(&kind, state, index)
            switch kind {
            case MLX_EXPORT_MIXED_BOOL:
                var v: Bool = false
                _ = mlx_export_state_mixed_scalar_array_bool(&v, state, index)
                out.append(.bool(v))
            case MLX_EXPORT_MIXED_INT:
                var v: Int32 = 0
                _ = mlx_export_state_mixed_scalar_array_int(&v, state, index)
                out.append(.int(v))
            case MLX_EXPORT_MIXED_FLOAT:
                var v: Float = 0
                _ = mlx_export_state_mixed_scalar_array_float(&v, state, index)
                out.append(.float(v))
            default:
                fatalError("Unknown mlx_export_mixed_scalar_kind — mlx version mismatch?")
            }
        }
        return .mixedScalarArray(out)
    case MLX_EXPORT_STATE_OPTIONAL_FLOAT:
        if mlx_export_state_optional_float_has_value(state) {
            var v: Float = 0
            _ = mlx_export_state_optional_float(&v, state)
            return .optionalFloat(v)
        }
        return .optionalFloat(nil)
    case MLX_EXPORT_STATE_STRING:
        let cstr = mlx_export_state_string(state)
        return .string(cstr.map { String(cString: $0) } ?? "")
    default:
        fatalError("Unknown mlx_export_state_kind — mlx version mismatch?")
    }
}

private func decodeStateVector<T>(
    _ state: mlx_export_state,
    _ getter: (
        UnsafeMutablePointer<UnsafePointer<T>?>?,
        UnsafeMutablePointer<Int>?,
        mlx_export_state
    ) -> Int32
) -> [T] {
    var data: UnsafePointer<T>?
    var size: Int = 0
    _ = getter(&data, &size, state)
    return data.map { ptr in
        Array(UnsafeBufferPointer(start: ptr, count: size))
    } ?? []
}

private func new_mlx_export_callback_closure(
    _ callback: @escaping (MLXExportCallbackPayload) -> Void
) -> mlx_closure_export_callback {
    func free(ptr: UnsafeMutableRawPointer?) {
        Unmanaged<CallbackCaptureState>.fromOpaque(ptr!).release()
    }
    let payload = Unmanaged.passRetained(CallbackCaptureState(callback))
        .toOpaque()

    func trampoline(
        input: mlx_export_callback_input,
        payload: UnsafeMutableRawPointer?
    ) {
        let state = Unmanaged<CallbackCaptureState>
            .fromOpaque(payload!).takeUnretainedValue()
        state.callback(decodeCallbackInput(input))
    }

    return mlx_closure_export_callback_new(trampoline, payload, free)
}

/// Imports a function from a `.mlxfn` file.
///
/// ```swift
/// // f is a callable that represents the loaded function
/// let f = try importFunction(from: url)
///
/// let a = MLXArray(10)
/// let b = MLXArray([5, 10, 20])
///
/// // call it -- the shapes and labels have to match
/// let r = try f(a, y: b)[0]
/// ```
///
/// - Parameter url: file to load from
/// - Returns: a callable that represents the loaded function
/// ### See Also
/// - ``exportFunction(to:shapeless:_:)``
/// - ``exportFunctions(to:shapeless:_:build:)``
public func importFunction(from url: URL) throws -> ImportedFunction {
    try ImportedFunction(url: url)
}

/// Helper for ``importFunction(from:)`` -- this holds the imported function.
///
/// This can be called with parameters that match the recorded parameters:
///
/// ```swift
/// func f(arrays: [MLXArray]) -> [MLXArray] {
///     [arrays[0] * arrays[1]]
/// }
///
/// let x = MLXArray(1)
/// let y = MLXArray([1, 2, 3])
///
/// // records with unnamed first parameter and a second parameter named `y`
/// try exportFunction(to: url, f)(x, y: y)
///
/// // f2 is a ImportedFunction
/// let f2 = try importFunction(from: url)
///
/// let a = MLXArray(10)
/// let b = MLXArray([5, 10, 20])
///
/// // call it -- the shapes and labels have to match
/// let r = try f2(a, y: b)[0]
/// ```
@dynamicCallable
public final class ImportedFunction {

    private let ctx: mlx_imported_function

    public init(url: URL) throws {
        self.ctx = try withError {
            mlx_imported_function_new(url.path)
        }
    }

    deinit {
        mlx_imported_function_free(ctx)
    }

    public func dynamicallyCall(withKeywordArguments args: KeyValuePairs<String, MLXArray>) throws
        -> [MLXArray]
    {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }

        let positionalArgs = mlx_vector_array_new()
        defer { mlx_vector_array_free(positionalArgs) }
        for (key, value) in args {
            if key.isEmpty {
                mlx_vector_array_append_value(positionalArgs, value.ctx)
            }
        }

        let kwargs = new_mlx_array_map(
            Dictionary(
                args.compactMap { $0.key.isEmpty ? nil : ($0.key, $0.value) },
                uniquingKeysWith: { a, b in a }))
        defer { mlx_map_string_to_array_free(kwargs) }

        _ = try withError {
            mlx_imported_function_apply_kwargs(&result, ctx, positionalArgs, kwargs)
        }

        return mlx_vector_array_values(result)
    }
}
