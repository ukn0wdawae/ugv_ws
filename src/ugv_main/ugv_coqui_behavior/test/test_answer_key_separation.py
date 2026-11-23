"""The robot side of the Coqui survey must never see the simulation's answer key.

The simulation side knows where the frogs are: the answer-key file ugv_audio/config/sim_truth/forest_survey.yaml,
the sound emitter's truth topics under /sim/, Gazebo's model/link states. The robot side (the survey mission,
the microphone-array localizer, the traversability node and their configs) may only use its own sensors.
This static check fails on any reference to the answer key or to simulation-only topics and services in
robot-side source. Files are collected when the test runs, so new robot-side files are covered automatically.

    cd src/ugv_main/ugv_coqui_behavior && python3 -m pytest test/test_answer_key_separation.py -q
"""
import glob
import os

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))      # src/ugv_main
FORBIDDEN = ('sim_truth', 'forest_survey.yaml', '/sim/', 'truth_file', 'active_frog_info', 'emitter_markers',
             'lock_active_caller', 'model_states', 'link_states', 'get_entity_state', 'gazebo_msgs')
# files in the robot-side package that are allowed to use the truth
EXEMPT = {
    'coqui_hunter_node.py': 'legacy compact-world node, reads /coqui/active_frog_info by design',
    'eval_survey.py': 'the grader: compares a finished survey with the truth file after the run',
}


def robot_side_files():
    beh = os.path.join(SRC, 'ugv_coqui_behavior')
    files = [p for p in glob.glob(os.path.join(beh, 'ugv_coqui_behavior', '**', '*.py'), recursive=True)
             if os.path.basename(p) not in EXEMPT]
    files += glob.glob(os.path.join(SRC, 'ugv_traversability', 'ugv_traversability', '**', '*.py'), recursive=True)
    files.append(os.path.join(SRC, 'ugv_audio', 'ugv_audio', 'mic_array_localizer.py'))
    for pattern in ('*.yaml', '*.yml'):
        files += glob.glob(os.path.join(beh, 'config', '**', pattern), recursive=True)
    return sorted(set(files))


def references(path):
    """'file:line: <forbidden substring> in: <source line>' for every forbidden substring in a file."""
    out = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for n, line in enumerate(f, 1):
            for bad in FORBIDDEN:
                if bad in line:
                    out.append(f'{os.path.relpath(path, SRC)}:{n}: {bad!r} in: {line.strip()}')
    return out


def test_robot_side_file_set_is_complete():
    files = robot_side_files()
    assert files, 'no robot-side files found'
    names = {os.path.basename(p) for p in files}
    for must in ('coqui_survey_node.py', 'survey_planning.py', 'mic_array_localizer.py', 'traversability_node.py',
                 'survey_base.yaml'):
        assert must in names, f'{must} is not scanned; scanned: {sorted(names)}'
    assert all(os.path.isfile(p) for p in files), [p for p in files if not os.path.isfile(p)]
    assert not names & set(EXEMPT)


def test_robot_side_never_references_the_answer_key():
    offences = [r for p in robot_side_files() for r in references(p)]
    assert not offences, ('robot-side code must not use the simulation answer key or simulation-only topics:\n'
                          + '\n'.join(offences))


def test_the_check_catches_a_reference(tmp_path):
    p = tmp_path / 'sneaky_node.py'
    p.write_text("x = 1\nsub = '/sim/coqui_truth_markers'   # where the frogs really are\n")
    found = references(str(p))
    assert len(found) == 1 and ':2:' in found[0] and "'/sim/'" in found[0]
