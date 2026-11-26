# azo-plugin-session-labeling

Userspace agent-zoo orchestrator plugin that adds `name_session` and system-prompt guidance for concise session labels.

Install with:

```bash
azo-plugin install /path/to/azo-plugin-session-labeling
```

## Runtime compatibility

This plugin requires `agent-utils>=0.4.0`, which provides the event-driven Sans-I/O pipeline and transient multi-channel render API (`RenderTransformRegistrar`, `MODEL_RENDER_CHANNEL`, and `Entry.register_render_transform`). The host loader raises an actionable compatibility error instead of silently skipping this plugin when that floor is unavailable.
