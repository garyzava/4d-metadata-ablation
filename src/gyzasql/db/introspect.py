"""Schema introspection for Postgres databases."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from sqlalchemy import inspect
from sqlalchemy.engine import Engine


class ColumnInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    data_type: str
    nullable: bool
    is_primary_key: bool = False


class ForeignKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_table: str
    from_column: str
    to_table: str
    to_column: str


class TableInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    columns: list[ColumnInfo] = []


class SchemaInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    tables: list[TableInfo] = []
    foreign_keys: list[ForeignKey] = []


def introspect_schema(engine: Engine, schema: str = "public") -> SchemaInfo:
    """Extract tables, columns, primary keys, and foreign keys from the database."""
    insp = inspect(engine)
    tables: list[TableInfo] = []
    foreign_keys: list[ForeignKey] = []

    for table_name in sorted(insp.get_table_names(schema=schema)):
        pk_cols = set(insp.get_pk_constraint(table_name, schema=schema).get("constrained_columns", []))
        columns = [
            ColumnInfo(
                name=col["name"],
                data_type=str(col["type"]),
                nullable=col.get("nullable", True),
                is_primary_key=col["name"] in pk_cols,
            )
            for col in insp.get_columns(table_name, schema=schema)
        ]
        tables.append(TableInfo(name=table_name, columns=columns))

        for fk in insp.get_foreign_keys(table_name, schema=schema):
            ref_table = fk["referred_table"]
            for local_col, remote_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                foreign_keys.append(
                    ForeignKey(
                        from_table=table_name,
                        from_column=local_col,
                        to_table=ref_table,
                        to_column=remote_col,
                    )
                )

    return SchemaInfo(tables=tables, foreign_keys=foreign_keys)
