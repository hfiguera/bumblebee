defmodule Bumblebee.Text.DebertaV2.DisentangledAttention do
  @moduledoc false
  import Nx.Defn

  defn project(hidden, relative, kernel, bias, opts) do
    opts = keyword!(opts, [:heads, :relative, :mode])
    content = split_heads(linear(hidden, kernel, bias), opts[:heads])

    position =
      if opts[:relative] do
        split_heads(linear(Nx.new_axis(relative, 0), kernel, bias), opts[:heads])
      else
        Nx.tensor(0.0)
      end

    {content, position}
  end

  defn weights(query, key, position_query, position_key, mask, opts) do
    opts = keyword!(opts, [:buckets, :max_position, :span, :relative, :terms, :mode])
    scale = Nx.sqrt(Nx.axis_size(query, -1) * (1 + term_count(opts[:terms])))
    scores = product(query, key / scale)

    scores =
      if opts[:relative] do
        positions =
          relative_positions(Nx.axis_size(query, 2), opts[:buckets], opts[:max_position])

        bias = Nx.broadcast(0, Nx.shape(scores))

        bias =
          if has_term?(opts[:terms], :content_to_position) do
            indices = Nx.clip(positions + opts[:span], 0, 2 * opts[:span] - 1)
            bias + gather(product(query, position_key), indices) / scale
          else
            bias
          end

        bias =
          if has_term?(opts[:terms], :position_to_content) do
            indices = Nx.clip(-positions + opts[:span], 0, 2 * opts[:span] - 1)

            bias +
              Nx.transpose(gather(product(key, position_query), indices), axes: [0, 1, 3, 2]) /
                scale
          else
            bias
          end

        scores + bias
      else
        scores
      end

    allowed = Nx.new_axis(Nx.new_axis(mask, 1), 2) * Nx.new_axis(Nx.new_axis(mask, 1), 3)

    scores =
      Nx.select(
        Nx.broadcast(allowed, Nx.shape(scores)),
        scores,
        Nx.Constants.min_finite(Nx.type(scores))
      )

    Axon.Activations.softmax(scores, axis: -1)
  end

  defn context(weights, value, _opts) do
    {batch, heads, length, width} = Nx.shape(value)

    Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, length, heads * width})
  end

  deftransformp(has_term?(terms, term), do: term in terms)
  deftransformp(term_count(terms), do: length(terms))

  defnp(linear(input, kernel, bias), do: Nx.dot(input, [-1], kernel, [0]) + bias)

  defn split_heads(input, heads) do
    {batch, length, hidden} = Nx.shape(input)

    input
    |> Nx.reshape({batch, length, heads, div(hidden, heads)})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  defnp product(left, right) do
    right =
      Nx.broadcast(
        right,
        {Nx.axis_size(left, 0), Nx.axis_size(right, 1), Nx.axis_size(right, 2),
         Nx.axis_size(right, 3)}
      )

    Nx.dot(left, [3], [0, 1], right, [3], [0, 1])
  end

  defnp gather(scores, positions) do
    shape =
      {Nx.axis_size(scores, 0), Nx.axis_size(scores, 1), Nx.axis_size(positions, 0),
       Nx.axis_size(positions, 1)}

    Nx.take_along_axis(scores, Nx.broadcast(positions, shape), axis: 3)
  end

  defn relative_positions(length, buckets, max_position) do
    ids = Nx.iota({length}, type: :s64)
    relative = Nx.new_axis(ids, 1) - Nx.new_axis(ids, 0)

    if buckets > 0 do
      mid = div(buckets, 2)
      absolute = Nx.select(Nx.abs(relative) < mid, mid - 1, Nx.abs(relative))

      logarithmic =
        Nx.ceil(Nx.log(absolute / mid) / Nx.log((max_position - 1) / mid) * (mid - 1)) + mid

      Nx.select(absolute <= mid, relative, logarithmic * Nx.sign(relative)) |> Nx.as_type(:s64)
    else
      relative
    end
  end
end
