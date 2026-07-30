import sys
import os
import time
import torch
import pytest

from config import ModelConfig
from data.tokenizer import HabitatTokenizer
from model.transformer import DeepSpaceSLM
from inference.engine import InferenceEngine


BENCHMARK_SENTENCES = [
    # Category A: Stock & Quantity Queries (30 sentences)
    "how many oxygen cans do we have in storage",
    "what is the current count of food rations in galley",
    "check stock level on medical kits",
    "do we have enough water filters remaining",
    "how many n2 tanks are left in bay a2",
    "report current quantity of air filters",
    "how many antibiotic packs are available in medical bay",
    "what is the stock status of co2 scrubbers",
    "check remaining quantity of freeze dried meals",
    "how many wrench sets are stored in gamma shelf",
    "count remaining eva suit patch kits in airlock",
    "how many battery cells do we have in reserve",
    "what is the stock level of solar panel modules",
    "how many sanitizer bottles are in waste storage",
    "check current supply of protein bars",
    "what is the count of valve actuators",
    "how many circuit boards are available for repair",
    "report stock level of seal gaskets",
    "how many tether cables do we have in airlock",
    "check availability of waste disposal bags",
    "how many syringe units are in medical cabinet",
    "what is the quantity of painkiller bottles",
    "how many bandage rolls are remaining",
    "check stock of water purifier cartridges",
    "how many soldering kits are in tool locker",
    "what is the current count of multimeters",
    "how many cable bundles are stored",
    "check remaining quantity of suit helmet visors",
    "how many pump assemblies do we have in reserve",
    "report quantity of vitamin supplement bottles",

    # Category B: Location & Storage Queries (30 sentences)
    "where can I find the wrench set",
    "locate the eva suit patch kit",
    "which locker holds the water filters",
    "where are the o2 canisters stored",
    "find the location of medical kits",
    "where is the n2 tank kept",
    "locate co2 scrubber in alpha locker",
    "where can crew find food rations",
    "which compartment has antibiotic packs",
    "where is the air filter stored",
    "locate pump assembly in spare parts",
    "where are seal gaskets located",
    "find storage location for circuit boards",
    "where are battery cells stored",
    "locate solar panel module in bay b2",
    "where is the tether cable kept",
    "locate suit helmet visor in airlock",
    "where can I find sanitizer bottles",
    "locate waste disposal bags in waste bin",
    "where is the multimeter stored",
    "find location of soldering kit",
    "where are cable bundles stored",
    "locate syringe units in medical bay",
    "where is the painkiller bottle kept",
    "locate bandage rolls in cabinet 1",
    "where can crew find freeze dried meals",
    "locate protein bars in galley storage",
    "where are vitamin supplements stored",
    "locate water purifier cartridges",
    "where is the primary valve actuator stored",

    # Category C: Forecast & Consumption Queries (30 sentences)
    "when will we run out of n2 tanks",
    "how long will oxygen canisters last at current rate",
    "what is the estimated consumption rate of food rations",
    "forecast water filter supply duration for 30 days",
    "when will medical kits expire or run out",
    "how many days remaining for co2 scrubbers",
    "forecast consumption of air filters over next week",
    "when do we need to restock antibiotic packs",
    "how long will freeze dried meals last the crew",
    "forecast usage rate of eva suit patch kits",
    "when will protein bar inventory reach critical low",
    "how long can we rely on current battery cells",
    "forecast consumption rate of sanitizer bottles",
    "when will we run out of waste disposal bags",
    "how long will painkiller supplies last",
    "forecast days remaining for bandage rolls",
    "when do we need to replace water purifier cartridges",
    "how long will vitamin supplements last crew of four",
    "forecast usage of seal gaskets during maintenance",
    "when will pump assembly stock reach zero",
    "how long will tether cables remain functional",
    "forecast consumption of suit helmet visors",
    "when do we need to restock circuit boards",
    "how long will solar panel modules last",
    "forecast usage rate of soldering kits",
    "when will multimeter calibration expire",
    "how long will cable bundles last during repair",
    "forecast daily usage of oxygen in habitat",
    "when will nitrogen pressure drop below threshold",
    "how long will co2 absorption efficiency remain high",

    # Category D: Maintenance & Action Queries (30 sentences)
    "record that engineer replaced air filter",
    "log maintenance on co2 scrubber in alpha locker",
    "inspected solar panel module on exterior hull",
    "repaired pump assembly in storage bay b1",
    "cleaned water purifier cartridge in galley",
    "calibrated multimeter in laboratory rack",
    "replaced seal gasket on primary airlock hatch",
    "inspected eva suit patch kit before deployment",
    "repaired circuit board in forward node bin",
    "logged maintenance on battery cell cluster",
    "replaced spent co2 scrubber with fresh unit",
    "inspected o2 canister pressure valve in bay a1",
    "repaired tether cable after eva operation",
    "cleaned suit helmet visor after spacewalk",
    "calibrated delta p sensor on ventilation HVAC",
    "replaced spent filter in galley water recycler",
    "inspected medical kit inventory before mission day",
    "repaired valve actuator on secondary feed line",
    "logged maintenance on solar panel articulation motor",
    "replaced damaged cable bundle in aft node bin",
    "inspected waste disposal system container",
    "calibrated thermometer in crew quarters cabinet",
    "repaired soldering kit connection in tool bay",
    "logged inspection of emergency trauma kit",
    "replaced spent antibiotic pack in medical cabinet",
    "inspected primary oxygen regulator in airlock",
    "repaired radiator panel coolant line",
    "calibrated voltmeter in engineering section",
    "logged replacement of primary seal gasket",
    "inspected life support environmental scrubber",

    # Category E: General Habitat & Conversational Queries (30 sentences)
    "what is the overall status of life support systems",
    "are there any critical stock alerts in the habitat",
    "report current environmental radiation level",
    "what is the mission day counter right now",
    "how many crew members are currently active",
    "is oxygen pressure within nominal safety bounds",
    "show all low inventory supply warnings",
    "what is the status of water recycler efficiency",
    "are there any active maintenance tasks pending",
    "report overall habitat supply status summary",
    "is co2 concentration below safety threshold",
    "what is the power output of solar panel modules",
    "are any medical supplies critically depleted",
    "report current radiation threat level from rap guard",
    "what is the status of primary airlock pressure seal",
    "are all spare parts accounted for in storage bay",
    "report total food ration count in galley",
    "what is the status of thermal radiator cooling",
    "are any tools missing from module gamma locker",
    "report status of eva suit patch kits",
    "is nitrogen tank pressure nominal across habitat",
    "what is the status of emergency medical kit 1",
    "are water filters functioning at peak absorption",
    "report status of backup battery storage cells",
    "what is the current status of waste recycling",
    "are there any alerts for expiring medical supplies",
    "report status of telemetry sensors in node alpha",
    "is habitat ventilation operating normally",
    "what is the status of crew quarters life support",
    "are all critical systems operational for mission day"
]


class TestNLPBenchmark:
    def test_150_sentence_nlp_benchmark(self):
        cfg = ModelConfig(
            vocab_size=256,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            ffn_dim=344,
            max_seq_len=64,
        )
        tokenizer = HabitatTokenizer(vocab_size=cfg.vocab_size)
        model = DeepSpaceSLM(cfg)

        ckpt_path = "checkpoints/model_trained.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

        engine = InferenceEngine(model, tokenizer, device="cpu")

        total_sentences = len(BENCHMARK_SENTENCES)
        total_tokens_generated = 0
        total_time_sec = 0.0

        for sentence in BENCHMARK_SENTENCES:
            formatted_prompt = f"<QUERY> {sentence}"
            t0 = time.time()
            output_text = engine.generate(formatted_prompt, max_new_tokens=32, greedy=True)
            dt = time.time() - t0

            tokens = tokenizer.encode(output_text, add_bos=False, add_eos=False)
            total_tokens_generated += len(tokens)
            total_time_sec += dt

            assert len(output_text) > 0

        avg_latency_ms = (total_time_sec / total_sentences) * 1000
        assert total_sentences == 150
        assert avg_latency_ms < 50.0, f"Average latency exceeded 50ms: {avg_latency_ms:.2f}ms"
