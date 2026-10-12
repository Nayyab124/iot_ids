
"""Basic tests for the project setup."""

from iot_ids import __version__
from iot_ids.config import load_config


def test_package_has_version():
    assert isinstance(__version__, str)
    assert __version__


def test_config_loads_and_ips_is_disabled():
    config = load_config()

    assert config["project"]["name"] == "IoT IDS/IPS"
    assert config["ips"]["enabled"] is False
    assert config["ips"]["mode"] == "alert_only"
