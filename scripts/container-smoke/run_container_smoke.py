"""Run offline CPU-only smoke checks against locally built advisor and cutoff images."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid


def command(argv, *, timeout=120, combine_output=False, **kwargs):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, **kwargs)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {argv!r}\n{result.stdout}\n{result.stderr}')
    return (result.stdout + (result.stderr if combine_output else "")).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--advisor-image', required=True)
    parser.add_argument('--cutoff-image', required=True)
    parser.add_argument('--docker-context')
    args = parser.parse_args()
    source = args.source.resolve()
    revision = command(['git', '-C', str(source), 'rev-parse', 'HEAD'])
    docker = ['docker'] + (['--context', args.docker_context] if args.docker_context else [])
    uid = uuid.uuid4().hex[:10]
    prefix = f'senpai-offline-smoke-{uid}'
    volumes = [f'{prefix}-{name}' for name in ('workspace', 'home', 'state')]
    container = f'{prefix}-advisor'
    with tempfile.TemporaryDirectory(prefix='senpai-offline-smoke-') as raw:
        files = Path(raw)
        files.chmod(0o755)
        shutil.copyfile(Path(__file__).with_name('container_smoke.py'), files / 'container_smoke.py')
        shutil.copyfile(Path(__file__).with_name('artifact_terminal_smoke.py'), files / 'artifact_terminal_smoke.py')
        with (files / 'source.tar').open('wb') as stream:
            subprocess.run(['git', '-C', str(source), 'archive', 'HEAD'], stdout=stream, check=True)
        (files / 'bin').mkdir()
        (files / 'bin' / 'gh').write_text('#!/bin/sh\n[ "$1 $2 $3" = "repo set-default smoke/target" ] || exit 97\n')
        (files / 'trusted-python').write_text(
            '#!/bin/sh\nif [ "$1 $2 $3" = "-P -m senpai_agent.supervisor" ]; then\n'
            '  exec /opt/senpai-venv/bin/python -P /smoke/container_smoke.py pid1\nfi\n'
            'exec /opt/senpai-venv/bin/python "$@"\n')
        (files / 'startup.sh').write_text(
            '#!/bin/bash\nset -euo pipefail\n'
            'export SENPAI_PROGRAM_CONTEXT_FILE=/state/program.b64\n'
            'export SENPAI_PROGRAM_SOURCE_COMMIT="$(cat /state/program-commit)"\n'
            'export SENPAI_PROGRAM_CONTENT_SHA256="$(cat /state/program-digest)"\n'
            'export SENPAI_LAUNCH_CONTEXT_B64="$(cat /state/launch.b64)"\n'
            'export SENPAI_PROGRAM_PATH=program.md\n'
            'exec /bin/bash /workspace/senpai/k8s/entrypoint-advisor.sh\n')
        for name in ('bin/gh', 'trusted-python', 'startup.sh'):
            (files / name).chmod(0o755)
        mounts = [
            '--mount', f'type=bind,source={files},target=/smoke,readonly',
            '--mount', f'type=volume,source={volumes[0]},target=/workspace',
            '--mount', f'type=volume,source={volumes[1]},target=/home/senpai',
            '--mount', f'type=volume,source={volumes[2]},target=/state',
        ]
        runtime = ['--network', 'none', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--pids-limit', '256', '--cpus', '2',
            '--memory', '2g', '--tmpfs', '/tmp:rw,nosuid,nodev,size=256m',
            '--tmpfs', '/var/lib/senpai:rw,nosuid,nodev,uid=10001,gid=10001,size=64m']
        try:
            for name in volumes:
                command(docker + ['volume', 'create', '--label', 'senpai.smoke=pr3515', name])
            command(docker + ['run', '--rm', '--network', 'none', '--user', '0', *mounts,
                args.advisor_image, '/opt/senpai-venv/bin/python', '-P', '/smoke/container_smoke.py', 'seed'])
            env = {
                'SENPAI_PYTHON':'/smoke/trusted-python',
                'PATH':'/smoke/bin:/opt/senpai-venv/bin:/home/senpai/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
                'HOME':'/home/senpai', 'RESEARCH_TAG':'offline-smoke', 'PROBLEM_DIR':'target',
                'ADVISOR_BRANCH':'advisor', 'STUDENT_NAMES':'synthetic-student', 'GH_REPO':'smoke/target',
                'WANDB_ENTITY':'synthetic', 'WANDB_PROJECT':'offline', 'NODES_PER_STUDENT':'1', 'GPUS_PER_STUDENT_NODE':'0',
                'LITELLM_LOCAL_MODEL_COST_MAP':'True', 'OPENHANDS_SUPPRESS_BANNER':'1',
                'SENPAI_REPO_URL':'https://github.com/smoke/runner.git', 'SENPAI_REPO_REVISION':revision,
                'TARGET_REPO_URL':'https://github.com/smoke/target.git', 'TARGET_REPO_BRANCH':'advisor',
                'GITHUB_TOKEN':'synthetic-github', 'EXA_API_KEY':'synthetic-exa', 'WANDB_API_KEY':'synthetic-wandb',
            }
            env_args = [item for name, value in env.items() for item in ('--env', f'{name}={value}')]
            command(docker + ['run', '-d', '--name', container, '--user', '10001:10001',
                *runtime, *mounts, *env_args, args.advisor_image, '/smoke/startup.sh'])
            for start in (1, 2):
                code = command(docker + ['wait', container], timeout=90)
                logs = command(docker + ['logs', container], combine_output=True)
                if code != '0':
                    raise AssertionError(f'advisor smoke start {start} exited {code}\n{logs}')
                assert f'"container_start": {start}' in logs and '"status": "ok"' in logs, logs
                print(f'advisor container start {start}: passed')
                if start == 1:
                    command(docker + ['run', '--rm', '--network', 'none', '--user', '10001:10001',
                        *mounts, args.advisor_image, '/opt/senpai-venv/bin/python', '-P',
                        '/smoke/container_smoke.py', 'poison'])
                    command(docker + ['start', container])
            print(logs)
            artifact = command(docker + ['run', '--rm', '--user', '10001:10001', *runtime, *mounts,
                args.advisor_image, '/opt/senpai-venv/bin/python', '-P', '/smoke/artifact_terminal_smoke.py', '/workspace/senpai'], timeout=90)
            assert '"result": "PASS"' in artifact and '"pooled_terminal": true' in artifact, artifact
            print(artifact)
            cutoff_smoke(docker, args.cutoff_image, source, files, mounts, runtime, revision)
        finally:
            subprocess.run(docker + ['rm', '-f', container], capture_output=True)
            for name in volumes:
                subprocess.run(docker + ['volume', 'rm', name], capture_output=True)


def cutoff_smoke(docker, image, source, files, mounts, runtime, revision):
    capture = files / 'capture-kubectl'
    capture.write_text('#!/bin/sh\nfor arg in "$@"; do\n case "$arg" in\n'
        ' --from-file=cutoff-job.sh=*) cp "${arg#--from-file=cutoff-job.sh=}" "$CAPTURED_CUTOFF_SCRIPT" ;;\n'
        ' esac\ndone\nprintf "apiVersion: v1\\nkind: ConfigMap\\n"\n')
    capture.chmod(0o755)
    (files / 'host-bin').mkdir(exist_ok=True)
    (files / 'host-bin' / 'python').symlink_to(sys.executable)
    environment = {**os.environ, 'KUBECTL':str(capture), 'CAPTURED_CUTOFF_SCRIPT':str(files / 'cutoff-job.sh'),
        'PATH':f"{files / 'host-bin'}:{os.environ['PATH']}"}
    command(['bash', str(source / 'scripts/arm_senpai_cluster_cutoff.sh'),
        '--run-slug', 'offline-cutoff', '--tags-csv', 'synthetic-cutoff',
        '--expected-pods', '1', '--expected-deployments', '1', '--budget-hours', '0',
        '--readiness-timeout-minutes', '0', '--image', f'local-cutoff:sha-{revision}', '--dry-run'], env=environment)
    (files / 'bin' / 'kubectl').write_text('''#!/usr/local/bin/python
import json,sys
from pathlib import Path
args=sys.argv[1:]
assert args[:2] == ['-n','synthetic-smoke'], args
args=args[2:]
with Path('/state/cutoff-calls.jsonl').open('a') as f: f.write(json.dumps(args)+'\\n')
if args[:2] == ['get','pods']:
    print(json.dumps({'items':[{'status':{'containerStatuses':[{'ready':True}]}}]}))
elif args[:2] == ['get','deployments']:
    print('synthetic-cutoff 1/1')
elif args[:2] == ['delete','deployments']:
    assert args[2:] == ['-l','research-tag in (synthetic-cutoff)','--ignore-not-found=true'], args
    print('synthetic deletion recorded')
else: raise SystemExit('unexpected kubectl operation')
''')
    (files / 'bin' / 'kubectl').chmod(0o755)
    version = command(docker + ['run', '--rm', '--network', 'none', '--read-only', '--user', '10001:10001', image,
        '/usr/local/bin/kubectl', 'version', '--client', '-o', 'json'])
    assert json.loads(version)['clientVersion']['gitVersion']
    env = {'PATH':'/smoke/bin:/usr/local/bin:/usr/bin:/bin', 'RUN_SLUG':'offline-cutoff',
        'TAGS_CSV':'synthetic-cutoff', 'EXPECTED_PODS':'1', 'EXPECTED_DEPLOYMENTS':'1',
        'READINESS_TIMEOUT_SECONDS':'0', 'BUDGET_SECONDS':'0', 'ARMING_DEADLINE_EPOCH':'1',
        'HARD_KILL_AT_EPOCH':'1', 'ARM_ID':'synthetic-arm', 'STATE_AUTH_KEY':'synthetic-authentication-key',
        'PVC_LOG_ROOT':'/state', 'START_GATE_PATH':'/state/gate', 'NAMESPACE':'synthetic-smoke'}
    env_args = [item for name,value in env.items() for item in ('--env',f'{name}={value}')]
    output = command(docker + ['run','--rm','--user','10001:10001',*runtime,*mounts,*env_args,image,
        '/bin/bash','/smoke/cutoff-job.sh'], timeout=20)
    assert 'synthetic deletion recorded' in output and 'Cluster cutoff job done' in output, output
    print('cutoff non-root/read-only-root execution: passed (synthetic kubectl; no cluster contact)')
    command(docker + ['run', '--rm', '--user', '10001:10001', *runtime, *mounts, image, 'python', '-c',
        'import os;from pathlib import Path; paths=[Path("/state/offline-cutoff/cutoff_state.json"),Path("/state/gate")];[(p.unlink(missing_ok=True),os.mkfifo(p)) for p in paths]'])
    fifo_output = command(docker + ['run','--rm','--user','10001:10001',*runtime,*mounts,*env_args,image,
        '/bin/bash','/smoke/cutoff-job.sh'], timeout=20)
    assert 'synthetic deletion recorded' in fifo_output and 'Cluster cutoff job done' in fifo_output, fifo_output
    print('cutoff Linux FIFO state/gate substitution: passed without blocking')


if __name__ == '__main__':
    main()
