# Project 3 : QUERY EXECUTION

In a relational database, SQL statements will be converted into logical query plans and converted into physical query plans after query optimization. The system completes the corresponding statement functions by executing the physical query plan. In this experiment, it is necessary to implement physical query plan execution functions for `bustub`, including sequential scanning, insertion, deletion, change, connection, aggregation, and `DISTINCT` and `LIMIT`.

## Query plan execution

![figure1](https://github.com/yunfanlong/Bustub-Query-Plan-Execution/blob/main/src/figure1.png)

In a relational database, the physical query plan is organized into a tree within the system and executed through a specific query processing model (iterator model, producer model). The model to be implemented in this experiment is an iterator model. As shown in the figure above, each query plan node of this model obtains the next tuple it needs through the `NEXT()` method until the `NEXT()` method Return false. In the execution flow, the `NEXT()` method of the root node is called first, and its control flow propagates downward to the leaf nodes.

In `bustub`, each query plan node `AbstractPlanNode` is included in the executor class `AbstractExecutor`. The user calls the `Next()` method and initialization `Init()` method of the query plan through the executor class. The query plan node stores the unique information required for the operation. For example, the sequential scan needs to save the table identifier to be scanned in the node, and the connection needs to save its child nodes and the predicate of the connection in the node. at the same time. The executor class also contains `ExecutorContext` context information, which represents the global information of the query plan, such as transactions, transaction managers, lock managers, etc.

## SeqScanExecutor

`SeqScanExecutor` performs a sequential scan operation, which sequentially traverses all tuples in its corresponding table through the `Next()` method and returns the tuples to the caller. In `bustub`, all table-related information is contained in `TableInfo`:

```C++
 40 struct TableInfo {
 41   /**
 42    * Construct a new TableInfo instance.
 43    * @param schema The table schema
 44    * @param name The table name
 45    * @param table An owning pointer to the table heap
 46    * @param oid The unique OID for the table
 47    */
 48   TableInfo(Schema schema, std::string name, std::unique_ptr<TableHeap> &&table, table_oid_t oid)
 49       : schema_{std::move(schema)}, name_{std::move(name)}, table_{std::move(table)}, oid_{oid} {    }  
 50   /** The table schema */
 51   Schema schema_;
 52   /** The table name */
 53   const std::string name_;
 54   /** An owning pointer to the table heap */
 55   std::unique_ptr<TableHeap> table_;
 56   /** The table OID */
 57   const table_oid_t oid_;
 58 };
```

The actual tuples in the table are stored in `TableHeap`, which contains all functional interfaces for inserting, searching, changing, and deleting tuples, and the tuples in it can be traversed sequentially through the `TableIterator` iterator. In `SeqScanExecutor`, add `TableInfo` and iterator private members for accessing table information and traversing the table. In `bustub`, all tables are saved in the directory `Catalog`, from which the corresponding `TableInfo` can be extracted through the table identifier:

```c++
SeqScanExecutor::SeqScanExecutor(ExecutorContext *exec_ctx, const SeqScanPlanNode *plan)
    : AbstractExecutor(exec_ctx),
      plan_(plan),
      iter_(nullptr, RID(INVALID_PAGE_ID, 0), nullptr),
      end_(nullptr, RID(INVALID_PAGE_ID, 0), nullptr) {
  table_oid_t oid = plan->GetTableOid();
  table_info_ = exec_ctx->GetCatalog()->GetTable(oid);
  iter_ = table_info_->table_->Begin(exec_ctx->GetTransaction());
  end_ = table_info_->table_->End();
}
```
In `Init()`, perform the initialization operations required for the plan node, and reset the iterator of the table here so that the query plan can re-traverse the table:

```C++
void SeqScanExecutor::Init() {
  iter_ = table_info_->table_->Begin(exec_ctx_->GetTransaction());
  end_ = table_info_->table_->End();
}
```

In `Next()`, plan nodes to traverse the table and return tuples via input parameters, returning false when the traversal is complete:

```C++
bool SeqScanExecutor::Next(Tuple *tuple, RID *rid) {
  const Schema *out_schema = this->GetOutputSchema();
  Schema table_schema = table_info_->schema_;
  while (iter_ != end_) {
    Tuple table_tuple = *iter_;
    *rid = table_tuple.GetRid();
    std::vector<Value> values;
    for (const auto &col : GetOutputSchema()->GetColumns()) {
      values.emplace_back(col.GetExpr()->Evaluate(&table_tuple, &table_schema));
    }
    *tuple = Tuple(values, out_schema);
    auto *predicate = plan_->GetPredicate();
    if (predicate == nullptr || predicate->Evaluate(tuple, out_schema).GetAs<bool>()) {
      ++iter_;
      return true;
    }
    ++iter_;
  }
  return false;
}
```

Here, the tuple is accessed through the iterator `iter_`. When the plan node predicate `predicate` is not empty, the `Evaluate` method of `predicate` is used to evaluate whether the current tuple satisfies the predicate. If it is satisfied, it is returned, otherwise it is traversed to the next one. tuple. It is worth noting that the tuples in the table should be reorganized in the `out_schema` mode. In `bustub`, the output tuples of all query plan nodes are constructed through various "`Evaluate`" methods in `ColumnValueExpression` of each column `Column` in `out_schema`, such as `Evaluate`, `EvaluateJoin`, `EvaluateAggregate`.

For a plan node with a specific `out_schema`, the way to construct the output tuple is to traverse the `Column` in `out_schema` and extract the row corresponding to the table tuple through the `Evaluate` method of `ColumnValueExpression` in `Column`:

```C++
 36   auto Evaluate(const Tuple *tuple, const Schema *schema) const -> Value override {
 37     return tuple->GetValue(schema, col_idx_);
 38   }
```

It can be seen that `Column` stores the column number of the column in the table schema, and `Evaluate` extracts the corresponding column from the table tuple based on the column number.

## InsertExecutor

In `InsertExecutor`, it inserts tuples into a specific table. The source of the tuples may be other plan nodes or a custom tuple array. Its specific source can be extracted through `IsRawInsert()`. In the constructor, extract the `TableInfo` of the table to be inserted, the tuple source, and all indexes in the table:

```c++
InsertExecutor::InsertExecutor(ExecutorContext *exec_ctx, const InsertPlanNode *plan,
                               std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan), child_executor_(child_executor.release()) {
  table_oid_t oid = plan->TableOid();
  table_info_ = exec_ctx->GetCatalog()->GetTable(oid);
  is_raw_ = plan->IsRawInsert();
  if (is_raw_) {
    size_ = plan->RawValues().size();
  }
  indexes_ = exec_ctx->GetCatalog()->GetTableIndexes(table_info_->name_);
}
```

In `Init`, when its tuple source is another plan node, the `Init()` method of the corresponding plan node is executed:

```C++
void InsertExecutor::Init() {
  if (!is_raw_) {
    child_executor_->Init();
  }
}
```

In `Next()`, different insertion strategies are implemented based on different tuple sources:

```C++
bool InsertExecutor::Next([[maybe_unused]] Tuple *tuple, RID *rid) {
  Transaction *txn = exec_ctx_->GetTransaction();
  Tuple tmp_tuple;
  RID tmp_rid;
  if (is_raw_) {
    for (uint32_t idx = 0; idx < size_; idx++) {
      const std::vector<Value> &raw_value = plan_->RawValuesAt(idx);
      tmp_tuple = Tuple(raw_value, &table_info_->schema_);
      if (table_info_->table_->InsertTuple(tmp_tuple, &tmp_rid, txn)) {
        for (auto indexinfo : indexes_) {
          indexinfo->index_->InsertEntry(
              tmp_tuple.KeyFromTuple(table_info_->schema_, indexinfo->key_schema_, indexinfo->index_->GetKeyAttrs()),
              tmp_rid, txn);
        }
      }
    }
    return false;
  }
  while (child_executor_->Next(&tmp_tuple, &tmp_rid)) {
    if (table_info_->table_->InsertTuple(tmp_tuple, &tmp_rid, txn)) {
      for (auto indexinfo : indexes_) {
        indexinfo->index_->InsertEntry(tmp_tuple.KeyFromTuple(*child_executor_->GetOutputSchema(),
                                                              indexinfo->key_schema_, indexinfo->index_->GetKeyAttrs()),
                                       tmp_rid, txn);
      }
    }
  }
  return false;
}
```

It should be noted that the `Insert` node should not output any tuples, so it always returns false, that is, all insertion operations should be completed in one `Next`. When the source is a custom tuple array, the corresponding tuples are constructed according to the table schema and inserted into the table; when the source is other plan nodes, all tuples are obtained through child nodes and inserted into the table. During the insertion process, all indexes in the table should be updated using `InsertEntry`, and the parameters of `InsertEntry` should be constructed by the `KeyFromTuple` method.

## UpdateExecutor and DeleteExecutor

`UpdateExecutor` and `DeleteExecutor` are used to update and delete tuples from a specific table. Their implementation methods are similar to `InsertExecutor`, but their tuple sources are only other plan nodes:

```C++

UpdateExecutor::UpdateExecutor(ExecutorContext *exec_ctx, const UpdatePlanNode *plan,
                               std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan), child_executor_(child_executor.release()) {
  table_oid_t oid = plan->TableOid();
  auto catalog = exec_ctx->GetCatalog();
  table_info_ = catalog->GetTable(oid);
  indexes_ = catalog->GetTableIndexes(table_info_->name_);
}

void UpdateExecutor::Init() { child_executor_->Init(); }

bool UpdateExecutor::Next([[maybe_unused]] Tuple *tuple, RID *rid) {
  Tuple src_tuple;
  auto *txn = this->GetExecutorContext()->GetTransaction();
  while (child_executor_->Next(&src_tuple, rid)) {
    *tuple = this->GenerateUpdatedTuple(src_tuple);
    if (table_info_->table_->UpdateTuple(*tuple, *rid, txn)) {
      for (auto indexinfo : indexes_) {
        indexinfo->index_->DeleteEntry(tuple->KeyFromTuple(*child_executor_->GetOutputSchema(), indexinfo->key_schema_,
                                                           indexinfo->index_->GetKeyAttrs()),
                                       *rid, txn);
        indexinfo->index_->InsertEntry(tuple->KeyFromTuple(*child_executor_->GetOutputSchema(), indexinfo->key_schema_,
                                                           indexinfo->index_->GetKeyAttrs()),
                                       *rid, txn);
      }
    }
  }
  return false;
}
```

In `UpdateExecutor::Next`, use the `GenerateUpdatedTuple` method to update the source tuple to a new tuple. When updating the index, delete all index records corresponding to the source tuple in the table, and add the index corresponding to the new tuple. Record.

```C++

DeleteExecutor::DeleteExecutor(ExecutorContext *exec_ctx, const DeletePlanNode *plan,
                               std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan), child_executor_(child_executor.release()) {
  table_oid_t oid = plan->TableOid();
  auto catalog = exec_ctx->GetCatalog();
  table_info_ = catalog->GetTable(oid);
  indexes_ = catalog->GetTableIndexes(table_info_->name_);
}

void DeleteExecutor::Init() { child_executor_->Init(); }

bool DeleteExecutor::Next([[maybe_unused]] Tuple *tuple, RID *rid) {
  auto *txn = this->GetExecutorContext()->GetTransaction();
  while (child_executor_->Next(tuple, rid)) {
    if (table_info_->table_->MarkDelete(*rid, txn)) {
      for (auto indexinfo : indexes_) {
        indexinfo->index_->DeleteEntry(tuple->KeyFromTuple(*child_executor_->GetOutputSchema(), indexinfo->key_schema_,
                                                           indexinfo->index_->GetKeyAttrs()),
                                       *rid, txn);
      }
    }
  }
  return false;
}
```

`DeleteExecutor` is similar to the previous two executors and will not be described again.

## NestedLoopJoinExecutor

`NestedLoopJoinExecutor` joins all tuples in the two child plan nodes, and each time it calls the `Next()` method it returns a connection tuple that conforms to the connection predicate to the parent node:

```C++

NestedLoopJoinExecutor::NestedLoopJoinExecutor(ExecutorContext *exec_ctx, const NestedLoopJoinPlanNode *plan,
                                               std::unique_ptr<AbstractExecutor> &&left_executor,
                                               std::unique_ptr<AbstractExecutor> &&right_executor)
    : AbstractExecutor(exec_ctx),
      plan_(plan),
      left_executor_(left_executor.release()),
      right_executor_(right_executor.release()) {}

void NestedLoopJoinExecutor::Init() {
  left_executor_->Init();
  right_executor_->Init();
  buffer_.clear();
  const Schema *left_schema = plan_->GetLeftPlan()->OutputSchema();
  const Schema *right_schema = plan_->GetRightPlan()->OutputSchema();
  const Schema *out_schema = this->GetOutputSchema();
  Tuple left_tuple;
  Tuple right_tuple;
  RID rid;
  while (left_executor_->Next(&left_tuple, &rid)) {
    right_executor_->Init();
    while (right_executor_->Next(&right_tuple, &rid)) {
      auto *predicate = plan_->Predicate();
      if (predicate == nullptr ||
          predicate->EvaluateJoin(&left_tuple, left_schema, &right_tuple, right_schema).GetAs<bool>()) {
        std::vector<Value> values;
        for (const auto &col : out_schema->GetColumns()) {
          values.emplace_back(col.GetExpr()->EvaluateJoin(&left_tuple, left_schema, &right_tuple, right_schema));
        }
        buffer_.emplace_back(values, out_schema);
      }
    }
  }
}
```

Here, the `Init()` function completes all connection operations and stores all the resulting connection tuples in the buffer `buffer_`. It obtains the tuple of the sub-plan node through the `Next()` method of the sub-plan node, traverses each pair of tuple combinations through a double-layer loop, and calls its `Init()` to initialize it when the inner plan node returns false. . After obtaining the subplan node tuple, if there is a predicate, call the predicate's EvaluateJoin to verify whether it meets the predicate. If there is no predicate or it matches the predicate, the output tuple is obtained by calling `EvaluateJoin` of each `Column` of `out_schema` and placed into `buffer_`.

```C++
bool NestedLoopJoinExecutor::Next(Tuple *tuple, RID *rid) {
  if (!buffer_.empty()) {
    *tuple = buffer_.back();
    buffer_.pop_back();
    return true;
  }
  return false;
}
```

In `Next()`, just extract the tuples in the buffer.

## HashJoinExecutor

`HashJoinExecutor` uses a basic hash join algorithm to perform a join operation. The principle is to use the join key of the tuple (that is, the combination of certain attribute columns) as the key of the hash table, and use the tuple of one of the sub-plan nodes to construct the hash Hope table. Since tuples with the same join key must have the same hash key value, a tuple in another subplan node only needs to look in the bucket of that tuple map for a tuple that can be joined to it, as shown in the following figure:

![figure2](https://github.com/yunfanlong/Bustub-Query-Plan-Execution/blob/main/src/figure2.png)

In order for the tuple to be inserted into the hash table, the corresponding hash function needs to be set for the connection key of the tuple, as well as the comparison method of its connection key:

```C++
struct HashJoinKey {
  Value value_;
  bool operator==(const HashJoinKey &other) const { return value_.CompareEquals(other.value_) == CmpBool::CmpTrue; }
};

}  // namespace bustub
namespace std {

/** Implements std::hash on AggregateKey */
template <>
struct hash<bustub::HashJoinKey> {
  std::size_t operator()(const bustub::HashJoinKey &key) const { return bustub::HashUtil::HashValue(&key.value_); }
};
```

For the hash function, use the built-in `HashUtil::HashValue` in `bustub`. Here, by reading the code, you can find that the connection key of the connection operation in `bustub` only consists of one attribute column of the tuple, so the specific value `Value` of a single attribute column only needs to be stored in the connection key, without the need for different Aggregation operations also store combinations of attribute columns `Vector<Value>`. Join keys are compared via `CompareEquals` of `Value`.

```C++
 private:
  /** The NestedLoopJoin plan node to be executed. */
  const HashJoinPlanNode *plan_;

  std::unique_ptr<AbstractExecutor> left_child_;

  std::unique_ptr<AbstractExecutor> right_child_;

  std::unordered_multimap<HashJoinKey, Tuple> hash_map_{};

  std::vector<Tuple> output_buffer_;
};

```
In `HashJoinExecutor`, use `unordered_multimap` to directly store tuples of connection keys. In principle, it is equivalent to the process of using an ordinary hash table and traversing buckets, but using `multimap` will make the implementation code more complex. Simple.
```C++

HashJoinExecutor::HashJoinExecutor(ExecutorContext *exec_ctx, const HashJoinPlanNode *plan,
                                   std::unique_ptr<AbstractExecutor> &&left_child,
                                   std::unique_ptr<AbstractExecutor> &&right_child)
    : AbstractExecutor(exec_ctx), plan_(plan), left_child_(left_child.release()), right_child_(right_child.release()) {}

void HashJoinExecutor::Init() {
  left_child_->Init();
  right_child_->Init();
  hash_map_.clear();
  output_buffer_.clear();
  Tuple left_tuple;
  const Schema *left_schema = left_child_->GetOutputSchema();
  RID rid;
  while (left_child_->Next(&left_tuple, &rid)) {
    HashJoinKey left_key;
    left_key.value_ = plan_->LeftJoinKeyExpression()->Evaluate(&left_tuple, left_schema);
    hash_map_.emplace(left_key, left_tuple);
  }
}
```

In `Init()`, `HashJoinExecutor` traverses the tuple of left subplan nodes to complete the hash table construction operation.

```C++

bool HashJoinExecutor::Next(Tuple *tuple, RID *rid) {
  if (!output_buffer_.empty()) {
    *tuple = output_buffer_.back();
    output_buffer_.pop_back();
    return true;
  }
  Tuple right_tuple;
  const Schema *left_schema = left_child_->GetOutputSchema();
  const Schema *right_schema = right_child_->GetOutputSchema();
  const Schema *out_schema = GetOutputSchema();
  while (right_child_->Next(&right_tuple, rid)) {
    HashJoinKey right_key;
    right_key.value_ = plan_->RightJoinKeyExpression()->Evaluate(&right_tuple, right_schema);
    auto iter = hash_map_.find(right_key);
    uint32_t num = hash_map_.count(right_key);
    for (uint32_t i = 0; i < num; ++i, ++iter) {
      std::vector<Value> values;
      for (const auto &col : out_schema->GetColumns()) {
        values.emplace_back(col.GetExpr()->EvaluateJoin(&iter->second, left_schema, &right_tuple, right_schema));
      }
      output_buffer_.emplace_back(values, out_schema);
    }
    if (!output_buffer_.empty()) {
      *tuple = output_buffer_.back();
      output_buffer_.pop_back();
      return true;
    }
  }
  return false;
}
```

In `Next()`, use the tuple of the right child plan node as a "probe" to find the tuple of the left child plan node with the same connection key as its connection key. It should be noted that a right node tuple may have multiple left node tuples corresponding to it, but a `Next()` operation only returns one result connection tuple. Therefore, in one `Next()`, all the result tuples obtained by the connection should be stored in the `output_buffer_` buffer, so that in the next `Next()`, only the result tuples need to be extracted from the buffer without calling The `Next()` method of child nodes.

## AggregationExecutor

`AggregationExecutor` implements aggregation operations. Its principle is to use a hash table to map all tuples with the same aggregation key together, so as to count the aggregation information of all aggregation key tuples:

```C++
 private:
  /** The aggregation plan node */
  const AggregationPlanNode *plan_;
  /** The child executor that produces tuples over which the aggregation is computed */
  std::unique_ptr<AbstractExecutor> child_;
  /** Simple aggregation hash table */
  // TODO(Student): Uncomment SimpleAggregationHashTable aht_;
  SimpleAggregationHashTable hash_table_;
  /** Simple aggregation hash table iterator */
  // TODO(Student): Uncomment SimpleAggregationHashTable::Iterator aht_iterator_;
  SimpleAggregationHashTable::Iterator iter_;
};
```

`SimpleAggregationHashTable` is a hash table used for aggregation operations. It uses the aggregation key (that is, the attribute column combination of the tuple) as the key, and uses the aggregated statistical result combination (`SUM`, `AVG` of the tuples with the same aggregation key , `MIN` and other statistics) as the value. It sets the iterator used to iterate over its elements.

```C++

AggregationExecutor::AggregationExecutor(ExecutorContext *exec_ctx, const AggregationPlanNode *plan,
                                         std::unique_ptr<AbstractExecutor> &&child)
    : AbstractExecutor(exec_ctx),
      plan_(plan),
      child_(child.release()),
      hash_table_(plan->GetAggregates(), plan->GetAggregateTypes()),
      iter_(hash_table_.Begin()) {}

void AggregationExecutor::Init() {
  Tuple tuple;
  RID rid;
  child_->Init();
  while (child_->Next(&tuple, &rid)) {
    hash_table_.InsertCombine(MakeAggregateKey(&tuple), MakeAggregateValue(&tuple));
  }
  iter_ = hash_table_.Begin();
}
```

In `Init()`, traverse the tuples of subplan nodes, build a hash table and set the iterator used to traverse the hash table. `InsertCombine` updates the statistics of the current aggregate key:

```c++
  void InsertCombine(const AggregateKey &agg_key, const AggregateValue &agg_val) {
    if (ht_.count(agg_key) == 0) {
      ht_.insert({agg_key, GenerateInitialAggregateValue()});
    }
    CombineAggregateValues(&ht_[agg_key], agg_val);
  }
```

```C++
  void CombineAggregateValues(AggregateValue *result, const AggregateValue &input) {
    for (uint32_t i = 0; i < agg_exprs_.size(); i++) {
      switch (agg_types_[i]) {
        case AggregationType::CountAggregate:
          // Count increases by one.
          result->aggregates_[i] = result->aggregates_[i].Add(ValueFactory::GetIntegerValue(1));
          break;
        case AggregationType::SumAggregate:
          // Sum increases by addition.
          result->aggregates_[i] = result->aggregates_[i].Add(input.aggregates_[i]);
          break;
        case AggregationType::MinAggregate:
          // Min is just the min.
          result->aggregates_[i] = result->aggregates_[i].Min(input.aggregates_[i]);
          break;
        case AggregationType::MaxAggregate:
          // Max is just the max.
          result->aggregates_[i] = result->aggregates_[i].Max(input.aggregates_[i]);
          break;
      }
    }
  }
```

In `Next()`, use an iterator to traverse the hash table. If there is a predicate, use the `EvaluateAggregate` of the predicate to determine whether the current aggregate key matches the predicate. If not, continue traversing until an aggregate key that matches the predicate is found.

```C++
bool AggregationExecutor::Next(Tuple *tuple, RID *rid) {
  while (iter_ != hash_table_.End()) {
    auto *having = plan_->GetHaving();
    const auto &key = iter_.Key().group_bys_;
    const auto &val = iter_.Val().aggregates_;
    if (having == nullptr || having->EvaluateAggregate(key, val).GetAs<bool>()) {
      std::vector<Value> values;
      for (const auto &col : GetOutputSchema()->GetColumns()) {
        values.emplace_back(col.GetExpr()->EvaluateAggregate(key, val));
      }
      *tuple = Tuple(values, GetOutputSchema());
      ++iter_;
      return true;
    }
    ++iter_;
  }
  return false;
}
```

## LimitExecutor

`LimitExecutor` is used to limit the number of output tuples, and the specific limit number is defined in its plan node. Its `Init()` should call the `Init()` method of the sub-plan node and reset the current limit quantity; the `Next()` method returns the tuple of the sub-plan node until the limit quantity is 0.

```C++

LimitExecutor::LimitExecutor(ExecutorContext *exec_ctx, const LimitPlanNode *plan,
                             std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan), child_executor_(child_executor.release()) {
  limit_ = plan_->GetLimit();
}

void LimitExecutor::Init() {
  child_executor_->Init();
  limit_ = plan_->GetLimit();
}

bool LimitExecutor::Next(Tuple *tuple, RID *rid) {
  if (limit_ == 0 || !child_executor_->Next(tuple, rid)) {
    return false;
  }
  --limit_;
  return true;
}
```

### DistinctExecutor

`DistinctExecutor` is used to remove identical input tuples and output different tuples. The hash table method is used here to remove duplicates. The specific construction strategy of the hash table refers to `SimpleAggregationHashTable` in the aggregation executor:

```C++
namespace bustub {
struct DistinctKey {
  std::vector<Value> value_;
  bool operator==(const DistinctKey &other) const {
    for (uint32_t i = 0; i < other.value_.size(); i++) {
      if (value_[i].CompareEquals(other.value_[i]) != CmpBool::CmpTrue) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace bustub

namespace std {

/** Implements std::hash on AggregateKey */
template <>
struct hash<bustub::DistinctKey> {
  std::size_t operator()(const bustub::DistinctKey &key) const {
    size_t curr_hash = 0;
    for (const auto &value : key.value_) {
      if (!value.IsNull()) {
        curr_hash = bustub::HashUtil::CombineHashes(curr_hash, bustub::HashUtil::HashValue(&value));
      }
    }
    return curr_hash;
  }
};
...

class DistinctExecutor : public AbstractExecutor {
...
  std::unordered_set<DistinctKey> set_;

  DistinctKey MakeKey(const Tuple *tuple) {
    std::vector<Value> values;
    const Schema *schema = GetOutputSchema();
    for (uint32_t i = 0; i < schema->GetColumnCount(); ++i) {
      values.emplace_back(tuple->GetValue(schema, i));
    }
    return {values};
  }
};
```

In actual operation, a hash table is used to remove duplicates. `Init()` clears the current hash table and initializes the sub-plan nodes. `Next()` determines whether the current tuple already appears in the hash table. If so, it traverses the next input tuple. If not, inserts the tuple into the hash table and returns:

```C++

DistinctExecutor::DistinctExecutor(ExecutorContext *exec_ctx, const DistinctPlanNode *plan,
                                   std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan), child_executor_(child_executor.release()) {}

void DistinctExecutor::Init() {
  set_.clear();
  child_executor_->Init();
}

bool DistinctExecutor::Next(Tuple *tuple, RID *rid) {
  while (child_executor_->Next(tuple, rid)) {
    auto key = MakeKey(tuple);
    if (set_.count(key) == 0U) {
      set_.insert(key);
      return true;
    }
  }
  return false;
}
```