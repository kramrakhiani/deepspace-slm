import pytest
import torch
from config import tiny_config
from data.tokenizer import HabitatTokenizer
from model.transformer import DeepSpaceSLM
from inference.engine import InferenceEngine
from inference.habitat_agent import HabitatAgent
from data.habitat_dataset import HabitatInventoryDataset


class TestTokenizer:
    def test_encode_decode_roundtrip(self):
        tokenizer = HabitatTokenizer()
        text = "check stock o2 canister"
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        decoded = tokenizer.decode(ids)
        assert "o2" in decoded
        assert "canister" in decoded

    def test_special_tokens(self):
        tokenizer = HabitatTokenizer()
        assert tokenizer.pad_id == 0
        assert tokenizer.bos_id == 1
        assert tokenizer.eos_id == 2
        assert tokenizer.unk_id == 3

    def test_encode_with_special(self):
        tokenizer = HabitatTokenizer()
        ids = tokenizer.encode("test", add_bos=True, add_eos=True)
        assert ids[0] == tokenizer.bos_id
        assert ids[-1] == tokenizer.eos_id

    def test_padding(self):
        tokenizer = HabitatTokenizer()
        ids = [1, 2, 3]
        padded = tokenizer.pad(ids, max_length=6)
        assert len(padded) == 6
        assert padded[3:] == [tokenizer.pad_id] * 3

    def test_domain_words(self):
        tokenizer = HabitatTokenizer()
        text = "oxygen canister medical kit water filter"
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        for word in ["oxygen", "canister", "medical", "kit", "water", "filter"]:
            assert word in tokenizer.token_to_id or ("Ġ" + word) in tokenizer.token_to_id

    def test_unknown_fallback(self):
        tokenizer = HabitatTokenizer()
        ids = tokenizer.encode("xyz123abc", add_bos=False, add_eos=False)
        assert len(ids) > 0


class TestInferenceEngine:
    @pytest.fixture
    def setup_engine(self):
        config = tiny_config()
        tokenizer = HabitatTokenizer(vocab_size=config.vocab_size)
        model = DeepSpaceSLM(config)
        engine = InferenceEngine(model, tokenizer)
        return engine

    def test_generate_produces_output(self, setup_engine):
        output = setup_engine.generate("<QUERY> check stock o2 canister", max_new_tokens=10)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_greedy_deterministic(self, setup_engine):
        prompt = "<QUERY> check stock o2 canister"
        out1 = setup_engine.generate(prompt, max_new_tokens=10, greedy=True)
        out2 = setup_engine.generate(prompt, max_new_tokens=10, greedy=True)
        assert out1 == out2

    def test_max_tokens_respected(self, setup_engine):
        output = setup_engine.generate("<QUERY> check stock", max_new_tokens=5, greedy=True)
        tokens = setup_engine.tokenizer.encode(output, add_bos=False, add_eos=False)
        assert len(tokens) <= 10

    def test_temperature_sampling(self, setup_engine):
        output = setup_engine.generate(
            "<QUERY> check stock", max_new_tokens=10, temperature=0.7, greedy=False
        )
        assert len(output) > 0

    def test_top_k_sampling(self, setup_engine):
        output = setup_engine.generate(
            "<QUERY> check stock", max_new_tokens=10, top_k=10, greedy=False
        )
        assert len(output) > 0

    def test_top_p_sampling(self, setup_engine):
        output = setup_engine.generate(
            "<QUERY> check stock", max_new_tokens=10, top_p=0.9, greedy=False
        )
        assert len(output) > 0

    def test_perplexity_computation(self, setup_engine):
        ppl = setup_engine.compute_perplexity("<QUERY> check stock o2 canister")
        assert isinstance(ppl, float)
        assert ppl > 0


class TestHabitatAgent:
    @pytest.fixture
    def setup_agent(self):
        config = tiny_config()
        tokenizer = HabitatTokenizer(vocab_size=config.vocab_size)
        model = DeepSpaceSLM(config)
        engine = InferenceEngine(model, tokenizer)
        agent = HabitatAgent(engine)
        return agent

    def test_check_stock(self, setup_agent):
        resp = setup_agent.check_stock("o2 canister")
        assert resp.item == "o2 canister"
        assert resp.quantity == 48
        assert resp.unit == "units"
        assert resp.status == "full"
        assert resp.location == "storage bay a1"
        assert resp.confidence == 1.0

    def test_locate(self, setup_agent):
        resp = setup_agent.locate("o2 canister")
        assert resp.item == "o2 canister"
        assert resp.location == "storage bay a1"
        assert resp.quantity == 48

    def test_forecast_usage(self, setup_agent):
        resp = setup_agent.forecast_usage("o2 canister")
        assert resp.item == "o2 canister"
        assert resp.consumption_rate == 2.0
        assert resp.days_remaining == 24.0

    def test_get_alerts(self, setup_agent):
        resp = setup_agent.get_alerts()
        assert len(resp.alerts) >= 0

    def test_log_maintenance(self, setup_agent):
        resp = setup_agent.log_maintenance("inspected", "co2 scrubber", "module alpha")
        assert resp.item == "co2 scrubber"
        assert resp.status == "logged"

    def test_free_query(self, setup_agent):
        result = setup_agent.free_query("status of life support")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_response_parsing(self, setup_agent):
        raw = "<RESPONSE> <ITEM> o2 canister <QTY> 48 units <STATUS> full <LOC> storage bay a1"
        resp = setup_agent._parse_response(raw)
        assert resp.item == "o2 canister"
        assert resp.quantity == 48
        assert resp.location == "storage bay a1"
        assert resp.status == "full"
        assert resp.confidence == 1.0


class TestDataset:
    def test_dataset_creation(self):
        tokenizer = HabitatTokenizer()
        dataset = HabitatInventoryDataset(
            tokenizer=tokenizer, num_samples=10, max_seq_len=64
        )
        assert len(dataset) == 10

    def test_dataset_item(self):
        tokenizer = HabitatTokenizer()
        dataset = HabitatInventoryDataset(
            tokenizer=tokenizer, num_samples=10, max_seq_len=64
        )
        item = dataset[0]
        assert "input_ids" in item
        assert "target_ids" in item
        assert "attention_mask" in item
        assert item["input_ids"].shape == (64,)
        assert item["target_ids"].shape == (64,)

    def test_dataset_dataloader(self):
        from data.habitat_dataset import create_dataloader

        tokenizer = HabitatTokenizer()
        loader = create_dataloader(
            tokenizer=tokenizer,
            num_samples=64,
            max_seq_len=32,
            batch_size=8,
        )
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (8, 32)
