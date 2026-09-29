"""Completed command logs are bounded without pruning research artifacts."""

import json
from pathlib import Path
import time
import uuid

import pytest

from senpai_agent.kubernetes_training import KubernetesTrainingSupervisor
from senpai_agent.training import KubernetesTrainingSpec, TrainingResult, TrainingSpec, TrainingState
from training_test_support import FakeCluster


def test_completed_log_history_survives_lost_local_state_and_partial_pruning(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    root = tmp_path / "outputs"
    monkeypatch.setenv("SENPAI_TRAINING_OUTPUT_ROOT", str(root))
    records = []
    for index in range(11):
        training_id = str(uuid.uuid4())
        output = root / training_id
        logs = output / ".senpai-logs"
        logs.mkdir(parents=True)
        with (logs / "node-0.log").open("wb") as stream:
            stream.truncate(63 * 1024 * 1024)
        (output / "checkpoint.pt").write_bytes(b"research result")
        (logs / "user-note.txt").write_text("not a managed log")
        result = TrainingResult(
            training_id=training_id, state=TrainingState.FINISHED,
            exit_code=0, elapsed_seconds=10 + 2 * index,
            log_path=str(state / f"{training_id}.log"),
            output_dir=str(output), started_at=100 - index,  # Completion order differs from start order.
            kubernetes_released=index != 10,
            worker_logs_pruned=index == 0,  # Crash after marking, before unlinking.
        )
        (state / f"{training_id}.json").write_text(result.model_dump_json())
        records.append(result)
    runtime = KubernetesTrainingSupervisor(
        workspace=tmp_path, state_dir=state, nodes=1, gpus_per_node=1,
    )
    try:
        for index, result in enumerate(records):
            output = Path(result.output_dir)
            assert (output / "checkpoint.pt").read_bytes() == b"research result"
            assert (output / ".senpai-logs/user-note.txt").is_file()
            assert (output / ".senpai-logs/node-0.log").exists() is (index >= 2)
            assert runtime.get_training_status(result.training_id).worker_logs_pruned is (index < 2)
            receipt = output / '.senpai-logs/released.json'
            assert receipt.exists() is (index != 10)
            if index != 10:
                assert json.loads(receipt.read_text()) == {
                    'released_at': result.started_at + result.elapsed_seconds,
                    'pruned': index < 2,
                }
    finally:
        runtime.close()

    # A replacement controller has no old local results. PVC release receipts
    # remain sufficient proof; an in-progress prune must finish after restart.
    for result in records[:-1]:
        (state / f'{result.training_id}.json').unlink()
    interrupted = Path(records[2].output_dir) / '.senpai-logs/released.json'
    receipt = json.loads(interrupted.read_text())
    receipt['pruned'] = True
    interrupted.write_text(json.dumps(receipt))
    for index in range(2):
        identifier = str(uuid.uuid4())
        output = root / identifier
        logs = output / '.senpai-logs'
        logs.mkdir(parents=True)
        with (logs / 'node-0.log').open('wb') as stream:
            stream.truncate(63 * 1024 * 1024)
        result = records[0].model_copy(update={
            'training_id': identifier, 'output_dir': str(output),
            'log_path': str(state / f'{identifier}.log'), 'started_at': 1000 + index,
            'worker_logs_pruned': False,
        })
        (state / f'{identifier}.json').write_text(result.model_dump_json())
    runtime = KubernetesTrainingSupervisor(
        workspace=tmp_path, state_dir=state, nodes=1, gpus_per_node=1,
    )
    try:
        assert not (interrupted.parent / 'node-0.log').exists()
        assert json.loads(interrupted.read_text())['pruned']
        oldest = Path(records[3].output_dir) / '.senpai-logs'
        assert not (oldest / 'node-0.log').exists()
        assert json.loads((oldest / 'released.json').read_text())['pruned']
        assert (Path(records[4].output_dir) / '.senpai-logs/node-0.log').exists()
        assert (Path(records[10].output_dir) / '.senpai-logs/node-0.log').exists()
        assert all((Path(result.output_dir) / 'checkpoint.pt').exists() for result in records)
    finally:
        runtime.close()


@pytest.mark.parametrize('linked_receipt', [False, True])
def test_unproven_history_blocks_growth_without_deleting_logs(tmp_path, monkeypatch, linked_receipt):
    root = tmp_path / 'outputs'
    monkeypatch.setenv('SENPAI_TRAINING_OUTPUT_ROOT', str(root))
    logs = root / str(uuid.uuid4()) / '.senpai-logs'
    logs.mkdir(parents=True)
    with (logs / 'node-0.log').open('wb') as stream:
        stream.truncate(512 * 1024 * 1024)
    (logs / 'released.tmp').write_text('interrupted receipt')
    checkpoint = logs.parent / 'checkpoint.pt'
    checkpoint.write_bytes(b'research result')
    outside = tmp_path / 'outside.json'
    outside.write_text('{"released_at": 1, "pruned": false}')
    if linked_receipt:
        (logs / 'released.json').symlink_to(outside)
    (root / str(uuid.uuid4())).symlink_to(logs.parent, target_is_directory=True)

    state = tmp_path / 'state'
    state.mkdir()
    identifier = str(uuid.uuid4())
    spec = KubernetesTrainingSpec(kind='Job', name='active-job', namespace='research', wandb_run_id=identifier)
    client = FakeCluster(TrainingState.RUNNING, nodes=1)
    client.reserve(identifier, spec, None, '/snapshot.bundle', 'a' * 40, nodes=1, gpus_per_node=1)
    active = TrainingResult(
        training_id=identifier, state=TrainingState.RUNNING, exit_code=None,
        elapsed_seconds=0, started_at=time.time(), log_path=str(state / f'{identifier}.log'),
        kubernetes_spec=spec, kubernetes_resource=client.resource_value, kubernetes_released=False,
        source_snapshot='/snapshot.bundle', source_commit='a' * 40, nodes=1, gpus_per_node=1,
    )
    (state / f'{identifier}.json').write_text(active.model_dump_json())
    runtime = KubernetesTrainingSupervisor(
        workspace=tmp_path, state_dir=state, nodes=1, gpus_per_node=1, poll_seconds=.01, client=client,
    )
    try:
        assert runtime.get_training_status(identifier).state is TrainingState.RUNNING
        with pytest.raises(RuntimeError, match='release receipt'):
            runtime.run_training(TrainingSpec(argv=('python', 'train.py'), cwd=tmp_path))
        assert client.deletions == []
        cancelled = runtime.cancel_training(identifier)
        assert cancelled.state is TrainingState.CANCELLED and cancelled.kubernetes_released
    finally:
        runtime.close()
    assert (logs / 'node-0.log').stat().st_size == 512 * 1024 * 1024
    assert (logs / 'released.tmp').read_text() == 'interrupted receipt'
    assert checkpoint.read_bytes() == b'research result'
    assert outside.read_text() == '{"released_at": 1, "pruned": false}'
