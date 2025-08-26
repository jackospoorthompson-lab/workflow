import os, glob, json, re, sys, subprocess, pathlib, yaml
from openai import OpenAI

ROOT = pathlib.Path(".")
PROMPT = os.environ["INPUT_PROMPT"].strip()
PATHS = [p.strip() for p in os.environ["INPUT_PATHS"].split(",") if p.strip()]
ALLOW_MAX_BYTES = 400_000  # guardrail: total file bytes
ALLOW_FILE_GLOBS = PATHS
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # override via repo secret/var if desired

# Optional repo policy file
policy_path = ROOT / ".ai-policy.yml"
policy = {}
if policy_path.exists():
    policy = yaml.safe_load(policy_path.read_text()) or {}
    ALLOW_MAX_BYTES = int(policy.get("max_bytes", ALLOW_MAX_BYTES))
    if "allow_globs" in policy:
        ALLOW_FILE_GLOBS = policy["allow_globs"]

def read_files():
    files = []
    seen = set()
    for g in ALLOW_FILE_GLOBS:
        for path in glob.glob(g, recursive=True):
            p = pathlib.Path(path)
            if not p.is_file():
                continue
            if p.suffix.lower() in {".png",".jpg",".jpeg",".gif",".pdf",".zip",".tar",".gz"}:
                continue
            if p.name.startswith(".ai-"):  # never edit config by accident
                continue
            if p in seen:
                continue
            seen.add(p)
            files.append(p)
    return files

files = read_files()
total_bytes = sum(p.stat().st_size for p in files)
if total_bytes > ALLOW_MAX_BYTES:
    print(f"Refusing: {total_bytes} bytes exceeds limit {ALLOW_MAX_BYTES}", file=sys.stderr)
    sys.exit(0)

def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()

# Capture original contents
original = {str(p): p.read_text(encoding="utf-8", errors="ignore") for p in files}

# Build a single instruction for structured edits
system_msg = """You are a precise repo editor. Apply the user's request to the provided files.
- Make minimal, high-quality edits.
- Preserve formatting and front-matter.
- Do not invent facts or break links.
- Return ONLY a JSON object with an array 'changes' of {path, content} for files that should be replaced entirely.
- If no changes needed, return {"changes": []}.
"""

# Construct a compact file bundle
file_bundle = "\n\n".join(
    f"=== FILE: {path} ===\n{content}"
    for path, content in original.items()
)

client = OpenAI()

response = client.responses.create(
    model=MODEL,
    reasoning={"effort":"medium"},
    input=[
        {"role":"system","content":system_msg},
        {"role":"user","content":f"User request:\n{PROMPT}\n\nProject files:\n{file_bundle}"}
    ],
    text_format= { "type": "json_schema", "json_schema": {
        "name":"changes_schema",
        "schema": {
            "type":"object",
            "properties":{
                "changes":{
                    "type":"array",
                    "items":{
                        "type":"object",
                        "properties":{
                            "path":{"type":"string"},
                            "content":{"type":"string"}
                        },
                        "required":["path","content"],
                        "additionalProperties": False
                    }
                }
            },
            "required":["changes"],
            "additionalProperties": False
        }
    }}
)

# Extract JSON from structured output
out_text = response.output_text
try:
    data = json.loads(out_text)
except Exception as e:
    print("Failed to parse model output as JSON; aborting.", file=sys.stderr)
    sys.exit(0)

changes = data.get("changes", [])
# Apply only to files in the allowlist
applied = 0
for ch in changes:
    p = pathlib.Path(ch["path"])
    # Must be within repo and in ALLOW_FILE_GLOBS
    if not any(p.match(gl) or p.as_posix().startswith(gl.rstrip("*")) for gl in ALLOW_FILE_GLOBS):
        continue
    if not p.exists():
        # create new files only if policy allows
        if not policy.get("allow_creates", True):
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
    # tidy line endings
    content = ch["content"].replace("\r\n","\n")
    p.write_text(content, encoding="utf-8")
    applied += 1

print(f"Applied changes to {applied} file(s).")
