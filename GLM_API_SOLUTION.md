
# 🚀 GLM API 429错误解决方案

## 问题分析
从日志可以看出，系统遇到了持续的429速率限制错误，即使添加了重试机制也无法解决。

## 可能的原因
1. **API配额耗尽** - 当前API密钥的请求配额可能已经用完
2. **速率限制过严** - API服务商设置了严格的并发请求限制
3. **账户权限问题** - 账户可能需要升级或重新配置

## 解决方案

### 方案1: 增加请求间隔
在 `modules/llm_director.py` 中修改:

```python
# 在类中添加
_min_request_interval = 5.0  # 增加到5秒间隔

# 在_request_llm方法开头添加
import time
current_time = time.time()
if hasattr(self, '_last_request_time'):
    time_diff = current_time - self._last_request_time
    if time_diff < self._min_request_interval:
        sleep_time = self._min_request_interval - time_diff
        logger.info(f"⏳ 遵守速率限制，等待 {sleep_time:.1f} 秒")
        time.sleep(sleep_time)
self._last_request_time = current_time
```

### 方案2: 实现队列处理
```python
import queue
import threading

class RequestQueue:
    def __init__(self):
        self.queue = queue.Queue()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
    
    def _worker(self):
        while True:
            func, args, kwargs = self.queue.get()
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"队列任务失败: {e}")
            finally:
                self.queue.task_done()
                time.sleep(2)  # 每个请求间隔2秒
    
    def add_request(self, func, *args, **kwargs):
        self.queue.put((func, args, kwargs))
```

### 方案3: 申请更高配额
联系智谱AI官方申请:
- 更高的RPM（每分钟请求数）限制
- 更大的TPM（每分钟token数）配额
- 商业账户升级

## 临时缓解措施

1. **降低处理速度**: 减少同时处理的章节数量
2. **增加重试延迟**: 将重试间隔增加到30-60秒
3. **批量处理**: 将多个小请求合并为一个大请求
4. **缓存结果**: 对已处理的内容进行缓存避免重复请求

## 监控建议

添加详细的日志记录:
```python
logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] API请求 - 章节: {chapter_name}, 字数: {len(content)}")
```
