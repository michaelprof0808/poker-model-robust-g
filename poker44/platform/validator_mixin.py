"""Platform/data-plane initialization shared by validator entrypoints."""

from poker44.platform.client import SubnetDataClient, SubnetDataConfig


class ValidatorPlatformMixin:
    def _initialize_platform_client(self) -> None:
        self.subnet_data_config = SubnetDataConfig.from_env()
        self.subnet_data = SubnetDataClient(self.subnet_data_config, self.wallet)
