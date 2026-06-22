"""Apply lazy import optimization to web_demo.py."""

with open("web_demo.py", encoding="utf-8") as f:
    content = f.read()

changes = 0

# 1. Replace top-level heavy imports with lightweight ones
old_top = (
    "from ACE_Agent.agent_core.supervisor import ACESupervisor\n"
    "from ACE_Agent.tools.data_factory import (\n"
    "    DATASET_LABELS,\n"
    "    generate_dataset,\n"
    "    list_demo_datasets,\n"
    "    load_custom_dataset,\n"
    ")\n"
    "from ACE_Agent.tools.llm_client import LLMSettings, UniversalLLMClient"
)
new_top = (
    "from ACE_Agent.tools.data_factory import (\n"
    "    DATASET_LABELS,\n"
    "    generate_dataset,\n"
    "    list_demo_datasets,\n"
    "    load_custom_dataset,\n"
    ")"
)
if old_top in content:
    content = content.replace(old_top, new_top)
    changes += 1
    print(f"[{changes}] Removed top-level heavy imports (ACESupervisor, llm_client)")
else:
    print("WARN: old_top pattern not found")

# 2. Lazy ACESupervisor in _get_supervisor
old_sup = (
    '@st.cache_resource(show_spinner="Initializing ACE Agent engine...")\n'
    "def _get_supervisor() -> ACESupervisor:\n"
    "    return ACESupervisor()"
)
new_sup = (
    '@st.cache_resource(show_spinner="Initializing ACE Agent engine...")\n'
    "def _get_supervisor():\n"
    "    from ACE_Agent.agent_core.supervisor import ACESupervisor as _ACE\n"
    "    return _ACE()"
)
if old_sup in content:
    content = content.replace(old_sup, new_sup)
    changes += 1
    print(f"[{changes}] Made _get_supervisor lazy")
else:
    print("WARN: old_sup pattern not found")

# 3. Lazy llm_client import in _sidebar_ui
# The sidebar uses DEFAULT_PROVIDERS, LLMSettings
# DEFAULT_PROVIDERS is already imported from settings_store
# LLMSettings now needs local import
old_sidebar = (
    '    """Render sidebar; return (primary_settings, fallback_settings)."""\n'
    "    sm = st.session_state.session_manager"
)
new_sidebar = (
    '    """Render sidebar; return (primary_settings, fallback_settings)."""\n'
    "    from ACE_Agent.tools.llm_client import LLMSettings as _LLMS\n"
    "    sm = st.session_state.session_manager"
)
if old_sidebar in content:
    content = content.replace(old_sidebar, new_sidebar)
    changes += 1
    print(f"[{changes}] Added lazy LLMSettings import to _sidebar_ui")
else:
    print("WARN: old_sidebar pattern not found")

# 4. Replace LLMSettings( references in _sidebar_ui with _LLMS(
if "_LLMS\n" in content or "_LLMS\r" in content or "_LLMS(" in content:
    # Only replace inside _sidebar_ui (between lines containing the import and the return)
    # Simpler: replace all LLMSettings( with _LLMS( - but that might be too aggressive
    # Let's just check if the function uses LLMSettings
    pass
# Actually, let me check what ghe sidebar uses
sidebar_section = content.find('def _sidebar_ui')
sidebar_end = content.find('\ndef _', sidebar_section + 100)
sidebar_text = content[sidebar_section:sidebar_end]
llms_count = sidebar_text.count('LLMSettings')
dp_count = sidebar_text.count('DEFAULT_PROVIDERS')
print(f"  Sidebar uses LLMSettings {llms_count} times, DEFAULT_PROVIDERS {dp_count} times")

# Replace all LLMSettings( in sidebar to _LLMS(
if llms_count > 0:
    before = content[:sidebar_section]
    after = content[sidebar_end:]
    sidebar_text = sidebar_text.replace('LLMSettings(', '_LLMS(')
    content = before + sidebar_text + after
    print(f"[{changes}] Replaced LLMSettings with _LLMS in _sidebar_ui")

# 5. Replace DEFAULT_PROVIDERS with local reference
if dp_count > 0:
    # DEFAULT_PROVIDERS is already imported from settings_store at top level
    # No change needed since settings_store is lightweight
    pass

# 6. Fix type annotations that reference removed types
# _sidebar_ui return type
old_ret = (
    "def _sidebar_ui() -> tuple[LLMSettings, LLMSettings | None]:"
)
new_ret = (
    "def _sidebar_ui():  # returns (LLMSettings, LLMSettings | None)"
)
if old_ret in content:
    content = content.replace(old_ret, new_ret)
    changes += 1
    print(f"[{changes}] Fixed _sidebar_ui return type annotation")
else:
    print("WARN: old_ret pattern not found")

# _accumulate_cost type hint
old_acc = (
    "def _accumulate_cost(client: UniversalLLMClient) -> None:"
)
new_acc = (
    "def _accumulate_cost(client) -> None:  # client: UniversalLLMClient"
)
if old_acc in content:
    content = content.replace(old_acc, new_acc)
    changes += 1
    print(f"[{changes}] Fixed _accumulate_cost type hint")
else:
    print("WARN: old_acc pattern not found")

# _handle_prompt signature
old_hp = (
    "    settings: LLMSettings,\n"
    "    fallback_settings: LLMSettings | None,"
)
new_hp = (
    "    settings,  # LLMSettings\n"
    "    fallback_settings,  # LLMSettings | None"
)
if old_hp in content:
    content = content.replace(old_hp, new_hp)
    changes += 1
    print(f"[{changes}] Fixed _handle_prompt signature")
else:
    print("WARN: old_hp pattern not found")

# 7. Lazy import of UniversalLLMClient in _handle_prompt
old_router = (
    '            router_client = UniversalLLMClient(settings, fallback_settings, caller="router")'
)
new_router = (
    '            from ACE_Agent.tools.llm_client import UniversalLLMClient as _LLMC\n'
    '            router_client = _LLMC(settings, fallback_settings, caller="router")'
)
if old_router in content:
    content = content.replace(old_router, new_router)
    changes += 1
    print(f"[{changes}] Made UniversalLLMClient lazy in _handle_prompt")
else:
    print("WARN: old_router pattern not found")

# 8. Add settings_store import for DEFAULT_PROVIDERS (already imported, verify)
# settings_store.py import at line 28 should already have DEFAULT_PROVIDERS
# Let's verify
if "from ACE_Agent.tools.settings_store import DEFAULT_PROVIDERS" in content:
    print("  DEFAULT_PROVIDERS import confirmed at top level (lightweight)")
else:
    print("  WARN: DEFAULT_PROVIDERS not in top-level imports")

with open("web_demo.py", "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nTotal changes applied: {changes}")
print("Done.")
