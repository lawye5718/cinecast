# CineCast 架构升级落实方案报告

## 一、尚未落实的架构升级（核对结果）

### 1. TTS 首句复读幻觉 Bug ✅ 已落实

**文件：** `modules/mlx_tts_engine.py`

**落实方案：**
- 第 17 行：已导入 `import re`
- 第 88-92 行：已实现标点符号补全防御逻辑：
  ```python
  render_text = content.strip()
  if not re.search(r'[。！？；.!?;]$', render_text):
      render_text += "。"
  ```
- 第 97-98 行：`model.generate()` 使用处理后的 `render_text` 而非原始 `content`：
  ```python
  results = list(self.model.generate(text=render_text, ...))
  ```

### 2. 全局选角字典与专属音色 ✅ 已落实

**文件：** `modules/asset_manager.py`

**落实方案：**
- 第 173-180 行：`get_voice_for_role` 方法中实现了角色专属音色文件匹配：
  ```python
  custom_voice_path = os.path.join(self.asset_dir, "voices", f"{speaker_name}.wav")
  if os.path.exists(custom_voice_path):
      self.role_voice_map[speaker_name] = {
          "audio": custom_voice_path,
          "text": f"角色专属音色 {speaker_name}",
          "speed": 1.0
      }
  ```
- 第 188-190 行：使用确定性哈希替代 `random.choice`：
  ```python
  digest = int(hashlib.md5(speaker_name.encode()).hexdigest(), 16)
  idx = digest % len(pool)
  self.role_voice_map[speaker_name] = pool[idx]
  ```

**文件：** `modules/llm_director.py`

**落实方案：**
- 第 138 行：`LLMScriptDirector.__init__` 已接收 `global_cast` 参数：
  ```python
  def __init__(self, ..., global_cast=None):
      self.global_cast = global_cast or {}
  ```
- 第 574-591 行：全局选角纪律注入已实现，将角色白名单追加到 system prompt

### 3. 极速试听模式与外部前情提要 ✅ 已落实

**文件：** `main_producer.py`

**落实方案：**
- 第 191 行：`phase_1_generate_scripts` 已接收 `is_preview` 参数
- 第 232-236 行：试听模式核心拦截逻辑（截断第一章前1000字）
- 第 252-253 行：读取 `custom_recaps` 字典逻辑
- 第 357-358 行：试听模式截断前10句逻辑
- 第 399-463 行：`run_preview_mode` 方法完整实现

---

## 二、本次审查新发现的逻辑冲突与优化点（已修复）

### 1. 逻辑冲突：合并后又切碎，做了无用功 ✅ 已优化

**文件：** `modules/llm_director.py`

**问题：** `parse_text_to_script` 方法末尾调用了 `merge_consecutive_narrators(full_script)`，将连续旁白合并为最大 800 字的长片段。但随后 `parse_and_micro_chunk` 方法又按 60 字进行微切片，使得合并操作完全被浪费。

**修复方案：** 移除 `parse_text_to_script` 中对 `merge_consecutive_narrators` 的调用（函数本身保留，以备其他场景使用），并添加注释说明原因：
```python
# 🌟 优化：移除 merge_consecutive_narrators 调用。
# 因为 parse_and_micro_chunk 会对结果进行严格的 60 字微切片，
# 合并后的 800 字长文本会被立即碾碎，属于无谓的算力浪费。
```

### 2. 插入位置的数组越界隐患（前情提要） ✅ 已优化

**文件：** `main_producer.py`（2 处）

**问题：** 原代码使用 `insert_idx = 1 if len(micro_script) > 0 else 0`。当 `micro_script` 只有 1 个元素时（例如极短章节或大模型幻觉导致只解析出一条内容），`insert_idx` 为 1，前情提要会被插入到唯一元素之后，而 `insert(insert_idx + 1, recap_unit)` 即 `insert(2, ...)` 会追加到末尾。虽然 Python 不会报错，但这种定位方式不够稳健——在极端边界情况下可能导致前情提要与内容的顺序不符合预期。

**修复方案：** 使用动态索引 `> 1` 替代 `> 0`（2 处均已修复）：
```python
insert_idx = 1 if len(micro_script) > 1 else 0
micro_script.insert(insert_idx, intro_unit)
micro_script.insert(insert_idx + 1, recap_unit)
```

---

## 三、测试验证

所有 176 个单元测试通过（包含新增的 3 个验证测试）：

- `TestMergeRemovedFromPipeline::test_parse_text_to_script_does_not_call_merge` - 验证 `parse_text_to_script` 不再调用 `merge_consecutive_narrators`
- `TestMergeRemovedFromPipeline::test_merge_function_still_importable` - 验证函数仍可导入
- `TestDynamicRecapInsertionIndex::test_source_uses_gt_1_guard` - 验证源码使用 `> 1` 而非 `> 0`
- `TestSafeRecapInsertion` 测试类（3 个测试）已更新为匹配新的 `> 1` 逻辑
