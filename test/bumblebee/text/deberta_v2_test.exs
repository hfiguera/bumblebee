defmodule Bumblebee.Text.DebertaV2Test do
  use ExUnit.Case, async: true
  import Nx.Testing
  import ExUnit.CaptureLog

  for variant <- [
        "absolute",
        "shared",
        "separate_conv",
        "content_to_position",
        "position_to_content"
      ] do
    @variant variant
    test "base encoder matches Python for #{@variant}" do
      directory = Path.expand("../../fixtures/deberta_v2/#{@variant}", __DIR__)
      reference = directory |> Path.join("expected.json") |> File.read!() |> Jason.decode!()

      log =
        capture_log(fn ->
          {:ok, info} = Bumblebee.load_model({:local, directory}, log_params_diff: true)
          assert info.spec.__struct__ == Bumblebee.Text.DebertaV2
          inputs = Map.new(reference["inputs"], fn {key, value} -> {key, Nx.tensor(value)} end)

          {_, predict} =
            Axon.build(info.model,
              compiler: EXLA,
              global_layer_options: [output_hidden_states: true, output_attentions: true]
            )

          result = predict.(info.params, inputs)

          assert_all_close(result.hidden_state, Nx.tensor(reference["hidden_state"]),
            atol: 2.0e-5,
            rtol: 2.0e-5
          )

          for {actual, expected} <-
                Enum.zip(Tuple.to_list(result.hidden_states), reference["hidden_states"]) do
            assert_all_close(actual, Nx.tensor(expected), atol: 2.0e-5, rtol: 2.0e-5)
          end

          for {actual, expected} <-
                Enum.zip(Tuple.to_list(result.attentions), reference["attentions"]) do
            assert_all_close(actual, Nx.tensor(expected), atol: 2.0e-5, rtol: 2.0e-5)
          end

          result = predict.(info.params, Map.take(inputs, ["input_ids"]))

          assert_all_close(result.hidden_state, Nx.tensor(reference["unmasked"]),
            atol: 2.0e-5,
            rtol: 2.0e-5
          )
        end)

      refute log =~ "were missing"
      refute log =~ "non-matching shape"
      refute log =~ "were unused"
    end
  end

  test "loads the DeBERTa Unigram tokenizer and special tokens" do
    directory = Path.expand("../../fixtures/deberta_v2/tokenizer", __DIR__)
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:local, directory})
    assert tokenizer.type == :deberta_v2
    expected = directory |> Path.join("expected.json") |> File.read!() |> Jason.decode!()
    actual = Bumblebee.apply_tokenizer(tokenizer, "hello world")
    assert_equal(actual["input_ids"], Nx.tensor([expected["input_ids"]]))
    assert_equal(actual["attention_mask"], Nx.tensor([expected["attention_mask"]]))
  end

  test "validates incompatible attention dimensions" do
    assert_raise ArgumentError, ~r/divisible/, fn ->
      Bumblebee.configure(Bumblebee.Text.DebertaV2, hidden_size: 13, num_attention_heads: 3)
    end
  end
end
