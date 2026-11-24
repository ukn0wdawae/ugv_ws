"""Shared pytest configuration & fixtures for the pooled ugv_ws /tests suite.

Pass custom navigation configs on the CLI to test ANY new navigation style:
  pytest tests/offline --nav2-params my_nav2.yaml --bt-xml my_bt.xml --traversability-params my_trav.yaml
  pytest tests/offline --style survey_dwb_default
"""
import os
import sys
import yaml
import pytest

WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_MAIN = os.path.join(WS_ROOT, 'src', 'ugv_main')

# Ensure robot packages are importable without sourcing ROS setup.bash
for pkg in ('ugv_coqui_behavior', 'ugv_traversability', 'ugv_audio'):
    p = os.path.join(SRC_MAIN, pkg)
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_addoption(parser):
    parser.addoption('--style', action='store', default='survey_dwb_default',
                     help='Named navigation style from tests/nav_styles.yaml')
    parser.addoption('--nav2-params', action='store', default='',
                     help='Override path to Nav2 YAML to test a new navigation style')
    parser.addoption('--bt-xml', action='store', default='',
                     help='Override path to Nav2 Behavior Tree XML')
    parser.addoption('--traversability-params', action='store', default='',
                     help='Override path to traversability YAML')
    parser.addoption('--survey-params', action='store', default='',
                     help='Override path to survey YAML')


def _resolve_path(path):
    if not path:
        return ''
    return path if os.path.isabs(path) else os.path.join(WS_ROOT, path)


@pytest.fixture(scope='session')
def ws_root():
    return WS_ROOT


@pytest.fixture(scope='session')
def nav_style(request):
    styles_doc = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'nav_styles.yaml')))
    style_name = request.config.getoption('--style')
    entry = dict(styles_doc['styles'][style_name])
    entry['name'] = style_name
    for cli_opt, key in (
        ('--nav2-params', 'nav2_params'),
        ('--bt-xml', 'bt_xml'),
        ('--traversability-params', 'traversability_params'),
        ('--survey-params', 'survey_params'),
    ):
        val = request.config.getoption(cli_opt)
        if val:
            entry[key] = val
    for key in ('nav2_params', 'bt_xml', 'traversability_params', 'survey_params'):
        entry[key] = _resolve_path(entry[key])
    return entry


@pytest.fixture(scope='session')
def nav2_cfg(nav_style):
    return yaml.safe_load(open(nav_style['nav2_params']))


@pytest.fixture(scope='session')
def trav_cfg(nav_style):
    return yaml.safe_load(open(nav_style['traversability_params']))


@pytest.fixture(scope='session')
def survey_cfg(nav_style):
    return yaml.safe_load(open(nav_style['survey_params']))


@pytest.fixture(scope='session')
def scenarios():
    return yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'cases', 'scenarios.yaml')))
