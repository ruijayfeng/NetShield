"""
Alert management system for network anomaly detection and cascading failure analysis.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import yaml
import os
from collections import defaultdict, deque
import threading
import time


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status states"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


class AlertCategory(Enum):
    """Categories of alerts"""
    ANOMALY_DETECTION = "anomaly_detection"
    CASCADING_FAILURE = "cascading_failure"
    NETWORK_HEALTH = "network_health"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE = "performance"


@dataclass
class AlertThresholds:
    """Configuration for alert thresholds"""
    info: float = 0.3
    warning: float = 0.6
    critical: float = 0.8
    emergency: float = 0.9
    
    @classmethod
    def from_config(cls, config_path: str = None):
        """Load thresholds from configuration file"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            thresholds = config.get('alerts', {}).get('thresholds', {})
            return cls(
                info=thresholds.get('info', 0.3),
                warning=thresholds.get('warning', 0.6),
                critical=thresholds.get('critical', 0.8),
                emergency=thresholds.get('emergency', 0.9)
            )
        except:
            return cls()


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    title: str
    description: str
    level: AlertLevel
    category: AlertCategory
    status: AlertStatus
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = None
    resolution_time: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledged_time: Optional[datetime] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization"""
        alert_dict = asdict(self)
        alert_dict['level'] = self.level.value
        alert_dict['category'] = self.category.value
        alert_dict['status'] = self.status.value
        alert_dict['timestamp'] = self.timestamp.isoformat()
        
        if self.resolution_time:
            alert_dict['resolution_time'] = self.resolution_time.isoformat()
        if self.acknowledged_time:
            alert_dict['acknowledged_time'] = self.acknowledged_time.isoformat()
        
        return alert_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create alert from dictionary"""
        alert = cls(
            id=data['id'],
            title=data['title'],
            description=data['description'],
            level=AlertLevel(data['level']),
            category=AlertCategory(data['category']),
            status=AlertStatus(data['status']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data['source'],
            metadata=data.get('metadata', {}),
            acknowledged_by=data.get('acknowledged_by'),
            tags=data.get('tags', [])
        )
        
        if data.get('resolution_time'):
            alert.resolution_time = datetime.fromisoformat(data['resolution_time'])
        if data.get('acknowledged_time'):
            alert.acknowledged_time = datetime.fromisoformat(data['acknowledged_time'])
        
        return alert


class AlertRule:
    """Rule for generating alerts based on conditions"""
    
    def __init__(self, rule_id: str, name: str, category: AlertCategory,
                 condition_func: Callable[[Dict[str, Any]], bool],
                 level_func: Callable[[Dict[str, Any]], AlertLevel],
                 title_func: Callable[[Dict[str, Any]], str],
                 description_func: Callable[[Dict[str, Any]], str]):
        self.rule_id = rule_id
        self.name = name
        self.category = category
        self.condition_func = condition_func
        self.level_func = level_func
        self.title_func = title_func
        self.description_func = description_func
        self.enabled = True
        self.last_triggered = None
        self.trigger_count = 0
    
    def evaluate(self, data: Dict[str, Any]) -> Optional[Alert]:
        """Evaluate rule against data and return alert if conditions met"""
        if not self.enabled:
            return None
        
        try:
            if self.condition_func(data):
                # Create alert
                alert_id = f"{self.rule_id}_{int(datetime.now().timestamp())}"
                
                alert = Alert(
                    id=alert_id,
                    title=self.title_func(data),
                    description=self.description_func(data),
                    level=self.level_func(data),
                    category=self.category,
                    status=AlertStatus.ACTIVE,
                    timestamp=datetime.now(),
                    source=self.rule_id,
                    metadata=data,
                    tags=[self.name, self.category.value]
                )
                
                self.last_triggered = datetime.now()
                self.trigger_count += 1
                
                return alert
        
        except Exception as e:
            logging.error(f"Error evaluating alert rule {self.rule_id}: {e}")
        
        return None


class NotificationChannel:
    """Base class for notification channels"""
    
    def __init__(self, channel_id: str, name: str):
        self.channel_id = channel_id
        self.name = name
        self.enabled = True
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert"""
        raise NotImplementedError


class ConsoleNotificationChannel(NotificationChannel):
    """Console notification channel"""
    
    def __init__(self):
        super().__init__("console", "Console Notifications")
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification to console"""
        try:
            level_emoji = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.CRITICAL: "🚨",
                AlertLevel.EMERGENCY: "🆘"
            }
            
            emoji = level_emoji.get(alert.level, "📢")
            print(f"\n{emoji} ALERT [{alert.level.value.upper()}] {emoji}")
            print(f"时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"标题: {alert.title}")
            print(f"描述: {alert.description}")
            print(f"来源: {alert.source}")
            print(f"分类: {alert.category.value}")
            if alert.tags:
                print(f"标签: {', '.join(alert.tags)}")
            print("-" * 50)
            
            return True
        
        except Exception as e:
            logging.error(f"Failed to send console notification: {e}")
            return False


class LogNotificationChannel(NotificationChannel):
    """Log file notification channel"""
    
    def __init__(self, log_file: str = "alerts.log"):
        super().__init__("log", "Log File Notifications")
        self.log_file = log_file
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup dedicated logger for alerts"""
        logger = logging.getLogger('alert_logger')
        logger.setLevel(logging.INFO)
        
        # Create handler if not exists
        if not logger.handlers:
            handler = logging.FileHandler(self.log_file, encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification to log file"""
        try:
            log_message = (
                f"[{alert.level.value.upper()}] {alert.title} | "
                f"Source: {alert.source} | Category: {alert.category.value} | "
                f"Description: {alert.description}"
            )
            
            if alert.level == AlertLevel.EMERGENCY:
                self.logger.critical(log_message)
            elif alert.level == AlertLevel.CRITICAL:
                self.logger.error(log_message)
            elif alert.level == AlertLevel.WARNING:
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)
            
            return True
        
        except Exception as e:
            logging.error(f"Failed to send log notification: {e}")
            return False


class AlertManager:
    """Main alert management system"""
    
    def __init__(self, thresholds: AlertThresholds = None):
        self.thresholds = thresholds or AlertThresholds()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.suppression_rules: List[Callable[[Alert], bool]] = []
        self.alert_stats = defaultdict(int)
        self.running = False
        
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize default notification channels
        self._initialize_default_channels()
        
        # Initialize built-in alert rules
        self._initialize_built_in_rules()
    
    def _initialize_default_channels(self):
        """Initialize default notification channels"""
        self.add_notification_channel(ConsoleNotificationChannel())
        self.add_notification_channel(LogNotificationChannel())
    
    def _initialize_built_in_rules(self):
        """Initialize built-in alert rules"""
        # Anomaly detection rule
        anomaly_rule = AlertRule(
            rule_id="anomaly_detection",
            name="网络异常检测",
            category=AlertCategory.ANOMALY_DETECTION,
            condition_func=lambda data: data.get('is_anomaly', False),
            level_func=self._get_anomaly_alert_level,
            title_func=lambda data: f"检测到网络异常 - 置信度 {data.get('confidence', 0):.2f}",
            description_func=self._get_anomaly_description
        )
        self.add_alert_rule(anomaly_rule)
        
        # Cascading failure rule
        cascade_rule = AlertRule(
            rule_id="cascading_failure",
            name="级联失效检测",
            category=AlertCategory.CASCADING_FAILURE,
            condition_func=lambda data: data.get('failure_ratio', 0) > 0.1,
            level_func=self._get_cascade_alert_level,
            title_func=lambda data: f"级联失效警告 - 失效比例 {data.get('failure_ratio', 0):.1%}",
            description_func=self._get_cascade_description
        )
        self.add_alert_rule(cascade_rule)
        
        # Network health rule
        health_rule = AlertRule(
            rule_id="network_health",
            name="网络健康监控",
            category=AlertCategory.NETWORK_HEALTH,
            condition_func=lambda data: data.get('health_score', 1.0) < 0.7,
            level_func=self._get_health_alert_level,
            title_func=lambda data: f"网络健康度下降 - 当前 {data.get('health_score', 0):.1%}",
            description_func=self._get_health_description
        )
        self.add_alert_rule(health_rule)
    
    def _get_anomaly_alert_level(self, data: Dict[str, Any]) -> AlertLevel:
        """Determine alert level for anomaly detection"""
        confidence = data.get('confidence', 0)
        
        if confidence >= self.thresholds.emergency:
            return AlertLevel.EMERGENCY
        elif confidence >= self.thresholds.critical:
            return AlertLevel.CRITICAL
        elif confidence >= self.thresholds.warning:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO
    
    def _get_anomaly_description(self, data: Dict[str, Any]) -> str:
        """Generate description for anomaly alert"""
        confidence = data.get('confidence', 0)
        node_id = data.get('node_id', '未知')
        features = data.get('top_features', [])
        
        desc = f"节点 {node_id} 检测到异常行为，置信度: {confidence:.3f}"
        
        if features:
            desc += f"\n主要异常特征: {', '.join(features[:3])}"
        
        return desc
    
    def _get_cascade_alert_level(self, data: Dict[str, Any]) -> AlertLevel:
        """Determine alert level for cascading failure"""
        failure_ratio = data.get('failure_ratio', 0)
        
        if failure_ratio >= 0.5:
            return AlertLevel.EMERGENCY
        elif failure_ratio >= 0.3:
            return AlertLevel.CRITICAL
        elif failure_ratio >= 0.2:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO
    
    def _get_cascade_description(self, data: Dict[str, Any]) -> str:
        """Generate description for cascading failure alert"""
        failure_ratio = data.get('failure_ratio', 0)
        failed_nodes = data.get('failed_nodes', 0)
        total_nodes = data.get('total_nodes', 0)
        iterations = data.get('iterations', 0)
        
        desc = f"检测到级联失效: {failed_nodes}/{total_nodes} 节点失效 ({failure_ratio:.1%})"
        desc += f"\n传播轮次: {iterations}"
        
        if 'initial_failures' in data:
            desc += f"\n初始失效节点: {', '.join(map(str, data['initial_failures']))}"
        
        return desc
    
    def _get_health_alert_level(self, data: Dict[str, Any]) -> AlertLevel:
        """Determine alert level for network health"""
        health_score = data.get('health_score', 1.0)
        
        if health_score < 0.3:
            return AlertLevel.EMERGENCY
        elif health_score < 0.5:
            return AlertLevel.CRITICAL
        elif health_score < 0.7:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO
    
    def _get_health_description(self, data: Dict[str, Any]) -> str:
        """Generate description for network health alert"""
        health_score = data.get('health_score', 0)
        metrics = data.get('metrics', {})
        
        desc = f"网络健康度降至 {health_score:.1%}"
        
        if metrics:
            issues = []
            if metrics.get('connectivity', 1.0) < 0.8:
                issues.append(f"连通性: {metrics['connectivity']:.1%}")
            if metrics.get('latency', 0) > 100:
                issues.append(f"延迟: {metrics['latency']:.1f}ms")
            if metrics.get('packet_loss', 0) > 0.05:
                issues.append(f"丢包率: {metrics['packet_loss']:.1%}")
            
            if issues:
                desc += f"\n问题指标: {'; '.join(issues)}"
        
        return desc
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add notification channel"""
        self.notification_channels[channel.channel_id] = channel
        self.logger.info(f"Added notification channel: {channel.name}")
    
    def remove_notification_channel(self, channel_id: str):
        """Remove notification channel"""
        if channel_id in self.notification_channels:
            channel = self.notification_channels.pop(channel_id)
            self.logger.info(f"Removed notification channel: {channel.name}")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove alert rule"""
        if rule_id in self.alert_rules:
            rule = self.alert_rules.pop(rule_id)
            self.logger.info(f"Removed alert rule: {rule.name}")
    
    def add_suppression_rule(self, rule_func: Callable[[Alert], bool]):
        """Add suppression rule"""
        self.suppression_rules.append(rule_func)
    
    async def create_alert(self, title: str, description: str, level: AlertLevel,
                          category: AlertCategory, source: str,
                          metadata: Dict[str, Any] = None,
                          tags: List[str] = None) -> Optional[str]:
        """Create a new alert manually"""
        
        alert_id = f"manual_{source}_{int(datetime.now().timestamp())}"
        
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            level=level,
            category=category,
            status=AlertStatus.ACTIVE,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {},
            tags=tags or []
        )
        
        return await self._process_alert(alert)
    
    async def evaluate_rules(self, data: Dict[str, Any]) -> List[str]:
        """Evaluate all rules against data and create alerts"""
        
        created_alerts = []
        
        for rule in self.alert_rules.values():
            alert = rule.evaluate(data)
            if alert:
                alert_id = await self._process_alert(alert)
                if alert_id:
                    created_alerts.append(alert_id)
        
        return created_alerts
    
    async def _process_alert(self, alert: Alert) -> Optional[str]:
        """Process and potentially create an alert"""
        
        # Check suppression rules
        if self._is_suppressed(alert):
            self.logger.debug(f"Alert suppressed: {alert.id}")
            return None
        
        # Check for duplicates (simple deduplication)
        if self._is_duplicate(alert):
            self.logger.debug(f"Duplicate alert detected: {alert.id}")
            return None
        
        # Add to active alerts
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Update statistics
        self.alert_stats[alert.level.value] += 1
        self.alert_stats[alert.category.value] += 1
        self.alert_stats['total'] += 1
        
        # Send notifications
        await self._send_notifications(alert)
        
        self.logger.info(f"Created alert: {alert.id} - {alert.title}")
        return alert.id
    
    def _is_suppressed(self, alert: Alert) -> bool:
        """Check if alert should be suppressed"""
        for rule in self.suppression_rules:
            try:
                if rule(alert):
                    return True
            except Exception as e:
                self.logger.error(f"Error in suppression rule: {e}")
        
        return False
    
    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate of recent alerts"""
        # Simple deduplication: check if similar alert exists in last 5 minutes
        cutoff_time = datetime.now() - timedelta(minutes=5)
        
        for existing_alert in self.active_alerts.values():
            if (existing_alert.timestamp > cutoff_time and
                existing_alert.title == alert.title and
                existing_alert.source == alert.source and
                existing_alert.category == alert.category):
                return True
        
        return False
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications through all channels"""
        
        notification_tasks = []
        
        for channel in self.notification_channels.values():
            if channel.enabled:
                task = asyncio.create_task(channel.send_notification(alert))
                notification_tasks.append(task)
        
        if notification_tasks:
            # Wait for all notifications to complete
            results = await asyncio.gather(*notification_tasks, return_exceptions=True)
            
            # Log any failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    channel_id = list(self.notification_channels.keys())[i]
                    self.logger.error(f"Notification failed for channel {channel_id}: {result}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_time = datetime.now()
            
            self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            alert.status = AlertStatus.RESOLVED
            alert.resolution_time = datetime.now()
            
            # Keep in history
            # (alert is already in alert_history)
            
            self.logger.info(f"Alert resolved: {alert_id}")
            return True
        
        return False
    
    def suppress_alert(self, alert_id: str) -> bool:
        """Suppress an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            
            self.logger.info(f"Alert suppressed: {alert_id}")
            return True
        
        return False
    
    def get_active_alerts(self, level: AlertLevel = None, 
                         category: AlertCategory = None) -> List[Alert]:
        """Get active alerts with optional filtering"""
        
        alerts = list(self.active_alerts.values())
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        if category:
            alerts = [alert for alert in alerts if alert.category == category]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return alerts
    
    def get_alert_statistics(self, time_range: timedelta = None) -> Dict[str, Any]:
        """Get alert statistics"""
        
        if time_range:
            cutoff_time = datetime.now() - time_range
            relevant_alerts = [alert for alert in self.alert_history 
                             if alert.timestamp >= cutoff_time]
        else:
            relevant_alerts = self.alert_history
        
        stats = {
            'total_alerts': len(relevant_alerts),
            'active_alerts': len(self.active_alerts),
            'resolved_alerts': len([alert for alert in relevant_alerts 
                                   if alert.status == AlertStatus.RESOLVED]),
            'by_level': defaultdict(int),
            'by_category': defaultdict(int),
            'by_source': defaultdict(int),
            'resolution_times': []
        }
        
        for alert in relevant_alerts:
            stats['by_level'][alert.level.value] += 1
            stats['by_category'][alert.category.value] += 1
            stats['by_source'][alert.source] += 1
            
            if alert.resolution_time and alert.status == AlertStatus.RESOLVED:
                resolution_time = (alert.resolution_time - alert.timestamp).total_seconds()
                stats['resolution_times'].append(resolution_time)
        
        # Convert defaultdict to regular dict
        stats['by_level'] = dict(stats['by_level'])
        stats['by_category'] = dict(stats['by_category'])
        stats['by_source'] = dict(stats['by_source'])
        
        # Calculate average resolution time
        if stats['resolution_times']:
            stats['average_resolution_time'] = sum(stats['resolution_times']) / len(stats['resolution_times'])
        else:
            stats['average_resolution_time'] = 0
        
        return stats
    
    def cleanup_old_alerts(self, max_age: timedelta = timedelta(days=7)):
        """Remove old alerts from history"""
        
        cutoff_time = datetime.now() - max_age
        
        # Keep recent alerts
        self.alert_history = [alert for alert in self.alert_history 
                            if alert.timestamp >= cutoff_time]
        
        self.logger.info(f"Cleaned up alerts older than {max_age}")
    
    def export_alerts(self, filepath: str, time_range: timedelta = None):
        """Export alerts to file"""
        
        if time_range:
            cutoff_time = datetime.now() - time_range
            alerts_to_export = [alert for alert in self.alert_history 
                              if alert.timestamp >= cutoff_time]
        else:
            alerts_to_export = self.alert_history
        
        # Convert to serializable format
        export_data = {
            'export_time': datetime.now().isoformat(),
            'total_alerts': len(alerts_to_export),
            'alerts': [alert.to_dict() for alert in alerts_to_export]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported {len(alerts_to_export)} alerts to {filepath}")
    
    def generate_alert_report(self, time_range: timedelta = timedelta(hours=24)) -> str:
        """Generate a comprehensive alert report"""
        
        stats = self.get_alert_statistics(time_range)
        
        report = []
        report.append("=" * 60)
        report.append("网络异常检测与级联失效分析 - 告警报告")
        report.append("=" * 60)
        
        # Time range info
        end_time = datetime.now()
        start_time = end_time - time_range
        report.append(f"报告时间范围: {start_time.strftime('%Y-%m-%d %H:%M')} 到 {end_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Summary statistics
        report.append(f"\n总体统计:")
        report.append(f"  总告警数量: {stats['total_alerts']}")
        report.append(f"  当前活跃告警: {stats['active_alerts']}")
        report.append(f"  已解决告警: {stats['resolved_alerts']}")
        
        if stats['average_resolution_time'] > 0:
            avg_resolution = stats['average_resolution_time'] / 60  # Convert to minutes
            report.append(f"  平均解决时间: {avg_resolution:.1f} 分钟")
        
        # By severity level
        report.append(f"\n按严重级别统计:")
        for level in ['emergency', 'critical', 'warning', 'info']:
            count = stats['by_level'].get(level, 0)
            if count > 0:
                report.append(f"  {level.upper()}: {count}")
        
        # By category
        report.append(f"\n按类别统计:")
        for category, count in stats['by_category'].items():
            report.append(f"  {category}: {count}")
        
        # Top sources
        report.append(f"\n主要告警来源:")
        sorted_sources = sorted(stats['by_source'].items(), key=lambda x: x[1], reverse=True)
        for source, count in sorted_sources[:5]:
            report.append(f"  {source}: {count}")
        
        # Current active alerts summary
        active_alerts = self.get_active_alerts()
        if active_alerts:
            report.append(f"\n当前活跃告警 (前5个):")
            for alert in active_alerts[:5]:
                report.append(f"  [{alert.level.value.upper()}] {alert.title}")
                report.append(f"    时间: {alert.timestamp.strftime('%H:%M:%S')}")
                report.append(f"    来源: {alert.source}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    async def test_alert_manager():
        # Create alert manager
        alert_manager = AlertManager()
        
        print("Testing Alert Manager...")
        
        # Test manual alert creation
        await alert_manager.create_alert(
            title="测试告警",
            description="这是一个测试告警",
            level=AlertLevel.WARNING,
            category=AlertCategory.SYSTEM_ERROR,
            source="test_system",
            tags=["test", "manual"]
        )
        
        # Test rule evaluation with anomaly data
        anomaly_data = {
            'is_anomaly': True,
            'confidence': 0.85,
            'node_id': 'node_001',
            'top_features': ['traffic', 'latency', 'cpu_usage']
        }
        
        created_alerts = await alert_manager.evaluate_rules(anomaly_data)
        print(f"Created {len(created_alerts)} alerts from rules")
        
        # Test cascading failure alert
        cascade_data = {
            'failure_ratio': 0.35,
            'failed_nodes': 7,
            'total_nodes': 20,
            'iterations': 3,
            'initial_failures': ['node_1', 'node_5']
        }
        
        await alert_manager.evaluate_rules(cascade_data)
        
        # Print statistics
        stats = alert_manager.get_alert_statistics()
        print(f"\nAlert Statistics: {stats}")
        
        # Generate report
        report = alert_manager.generate_alert_report()
        print(f"\nAlert Report:\n{report}")
        
        # Test alert acknowledgment and resolution
        active_alerts = alert_manager.get_active_alerts()
        if active_alerts:
            first_alert = active_alerts[0]
            alert_manager.acknowledge_alert(first_alert.id, "test_user")
            alert_manager.resolve_alert(first_alert.id)
            print(f"Acknowledged and resolved alert: {first_alert.id}")
    
    # Run test
    asyncio.run(test_alert_manager())