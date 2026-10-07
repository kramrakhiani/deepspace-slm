#!/usr/bin/env python3
"""
DeepSpace-SLM Interactive Terminal CLI
=========================================
Multi-agent command center for Space Exploration Logistics & Infrastructure SLM.
Routes commands through the AgentCoordinator to domain-specific agents.
Supports toggling between Conversational Prose Mode and Machine Structured Mode.
"""

import sys
import os
import torch
from config import ModelConfig, base_config, tiny_config
from data.tokenizer import HabitatTokenizer
from model.transformer import DeepSpaceSLM
from inference.engine import InferenceEngine
from inference.habitat_agent import HabitatAgent
from inference.agents.base import AgentMessage, MessagePriority
from quantization.rapg_engine import RadiationThreatLevel


def print_banner():
    banner = """
\033[36m  ____  _____ _____ ____  ____  ____   _     ____ _____ ____    ____  _     __  __ 
 |  _ \| ____| ____|  _ \/ ___||  _ \ / \   / ___| ____/ ___|  / ___|| |   |  \/  |
 | | | |  _| |  _| | |_) \___ \| |_) / _ \ | |   |  _| \___ \  \___ \| |   | |\/| |
 | |_| | |___| |___|  __/ ___) |  __/ ___ \| |___| |___ ___) |  ___) | |___| |  | |
 |____/|_____|_____|_|   |____/|_| /_/   \_\____|_____|____/  |____/|_____|_|  |_|\033[0m
 \033[33m--- Multi-Agent Space Logistics & Infrastructure Command Center (v3.0) ---\033[0m
    """
    print(banner)


def print_help():
    print("\033[1mAVAILABLE COMMANDS:\033[0m")
    print("  \033[32m/mode\033[0m              Toggle output format (Conversational Prose vs Machine Structured)")
    print("  \033[32m/stock <item>\033[0m      Check current inventory stock & status (e.g. /stock o2 canister)")
    print("  \033[32m/locate <item>\033[0m     Find item storage location (e.g. /locate medical kit)")
    print("  \033[32m/forecast <item>\033[0m   Estimate supply duration & consumption rate")
    print("  \033[32m/alerts\033[0m            Display critical & low stock warnings")
    print("  \033[32m/manual <query>\033[0m    Search habitat procedure manual (e.g. /manual co2 scrubber)")
    print("  \033[32m/rad <value>\033[0m       Simulate space radiation uGy/h (e.g. /rad 1200 for solar flare)")
    print("  \033[32m/status\033[0m            Display full habitat status report")
    print("  \033[32m/agents\033[0m            List all registered agents and their capabilities")
    print("  \033[32m/mesh\033[0m              Display DTN mesh network status")
    print("  \033[32m/audit\033[0m             Verify Merkle DAG audit chain integrity")
    print("  \033[32m/help\033[0m              Show this help menu")
    print("  \033[32m/exit\033[0m or \033[32m/quit\033[0m     Exit the CLI")
    print("\n  \033[35mOr type any free-form question to query the DeepSpace-SLM inference engine directly!\033[0m\n")


def print_agents(agent):
    """Display all registered agents and their capabilities."""
    agents = agent.coordinator.registered_agents
    print(f"\033[36m╔══════════════════════════════════════════════════╗\033[0m")
    print(f"\033[36m║  🤖 REGISTERED AGENTS ({len(agents)})                        ║\033[0m")
    print(f"\033[36m╠══════════════════════════════════════════════════╣\033[0m")
    for aid, ag in agents.items():
        caps = ag.capabilities
        print(f"\033[36m║\033[0m  \033[1;33m{aid:<15s}\033[0m  {len(caps)} capabilities")
        for cap in caps:
            print(f"\033[36m║\033[0m    \033[32m• {cap.intent:<20s}\033[0m {cap.description}")
    print(f"\033[36m╚══════════════════════════════════════════════════╝\033[0m")


def main():
    print_banner()
    print("\033[90mInitializing DeepSpace-SLM Multi-Agent System...\033[0m")

    cfg = ModelConfig(
        vocab_size=256,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        ffn_dim=344,
        max_seq_len=64,
    )
    tok = HabitatTokenizer(vocab_size=cfg.vocab_size)
    model = DeepSpaceSLM(cfg)

    ckpt_path = "checkpoints/model_trained.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\033[32m[TRAINED WEIGHTS LOADED] Loaded PyTorch SLM weights from {ckpt_path}\033[0m")

    engine = InferenceEngine(model, tok)
    agent = HabitatAgent(engine, prose_mode=True)

    # Show registered agents
    agent_count = len(agent.coordinator.registered_agents)
    print(f"\033[32m[MULTI-AGENT SYSTEM NOMINAL] {agent_count} domain agents active. Conversational Prose Mode.\033[0m\n")
    print_help()

    while True:
        try:
            user_input = input("\033[1;34mDeepSpace-SLM > \033[0m").strip()
            if not user_input:
                continue

            cmd_lower = user_input.lower()

            if cmd_lower in ("/exit", "/quit", "exit", "quit"):
                print("\033[33mTerminating DeepSpace-SLM session. Safe space operations!\033[0m")
                sys.exit(0)

            elif cmd_lower in ("/help", "help"):
                print_help()

            elif cmd_lower == "/mode":
                agent.set_prose_mode(not agent.prose_mode)
                mode_str = "Conversational Prose" if agent.prose_mode else "Machine Structured"
                print(f"\033[33m[Output Format Switched to: {mode_str}]\033[0m")

            elif cmd_lower == "/status":
                print(agent.status_report())

            elif cmd_lower == "/agents":
                print_agents(agent)

            elif cmd_lower == "/alerts":
                resp = agent.get_alerts()
                print(f"\033[33m{resp.raw_text}\033[0m")

            elif cmd_lower == "/audit":
                response = agent.coordinator.dispatch(AgentMessage(
                    intent="verify_audit", payload={}, source="cli",
                ))
                for msg in response.messages:
                    color = "\033[32m" if response.data.get("audit_valid", False) else "\033[31m"
                    print(f"{color}{msg}\033[0m")

            elif cmd_lower == "/mesh":
                response = agent.coordinator.dispatch(AgentMessage(
                    intent="mesh_status", payload={}, source="cli",
                ))
                print(f"\033[36m[DTN MESH STATUS]\033[0m")
                for msg in response.messages:
                    print(f"  {msg}")

            elif cmd_lower.startswith("/stock "):
                item = user_input[7:].strip()
                resp = agent.check_stock(item)
                print(f"\033[36m[STOCK REPORT]\033[0m {resp.raw_text}")

            elif cmd_lower.startswith("/locate "):
                item = user_input[8:].strip()
                resp = agent.locate(item)
                print(f"\033[36m[LOCATION]\033[0m {resp.raw_text}")

            elif cmd_lower.startswith("/forecast "):
                item = user_input[10:].strip()
                resp = agent.forecast_usage(item)
                print(f"\033[36m[FORECAST]\033[0m {resp.raw_text}")

            elif cmd_lower.startswith("/manual "):
                query = user_input[8:].strip()
                results = agent.query_procedure_manual(query)
                if results:
                    print(f"\033[36m[MANUAL RAG MATCHES ({len(results)})]\033[0m")
                    for r in results:
                        print(f"  \033[1m• {r['title']}\033[0m ({r['topic']}):")
                        print(f"    {r['content']}\n")
                else:
                    print("\033[31mNo matching procedure manual entry found.\033[0m")

            elif cmd_lower.startswith("/rad "):
                try:
                    rad_val = float(user_input[5:].strip())
                    status = agent.process_environmental_telemetry(rad_val)
                    color = "\033[32m" if status.threat_level == RadiationThreatLevel.NOMINAL else ("\033[33m" if status.threat_level == RadiationThreatLevel.ELEVATED else "\033[31m")
                    print(f"{color}[RAP-G DYNAMIC DEFENSE UPDATE]\033[0m")
                    print(f"  Radiation Level: {status.radiation_value_ugy_h:.1f} uGy/h")
                    print(f"  Threat Level:    {status.threat_level.name}")
                    print(f"  Active Defenses: {', '.join(status.active_defenses)}")
                except ValueError:
                    print("\033[31mInvalid radiation float value. Example: /rad 1200\033[0m")

            else:
                # Free-form SLM query — routed through InferenceAgent via coordinator
                output = agent.free_query(user_input)
                print(f"\033[32m[SLM RESPONSE]\033[0m {output}")

        except (KeyboardInterrupt, EOFError):
            print("\n\033[33mSession interrupted. Goodbye!\033[0m")
            sys.exit(0)


if __name__ == "__main__":
    main()
