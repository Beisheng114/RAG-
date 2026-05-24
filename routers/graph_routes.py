from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Form, UploadFile, File
from pydantic import BaseModel

from services.graph_service import (
    query_graph_data,
    get_node_counts_data,
    preview_material_data,
    import_material_with_parsed_data,
    import_material_data,
)


router = APIRouter(prefix="/api", tags=["graph"])


class GraphQueryRequest(BaseModel):
    query: str
    entity_type: str = "all"
    node_limit: int = 200
    system_name: Optional[str] = "all"


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    title: Optional[str] = None
    system_name: Optional[str] = None


class GraphEdge(BaseModel):
    from_id: str
    to_id: str
    label: str


class GraphQueryResponse(BaseModel):
    success: bool
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    stats: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class NodeCountsResponse(BaseModel):
    success: bool
    counts: Optional[Dict[str, int]] = None
    message: Optional[str] = None


class MaterialImportResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


@router.post("/graph/query", response_model=GraphQueryResponse)
def query_graph(request: GraphQueryRequest):
    try:
        data = query_graph_data(request.query, request.entity_type, request.node_limit, request.system_name or "all")
        if not data["success"]:
            return GraphQueryResponse(success=False, nodes=[], edges=[], stats=None, message=data.get("message"))

        graph_nodes = [
            GraphNode(
                id=node["id"],
                label=node["label"],
                type=node["type"],
                title=node.get("title"),
                system_name=node.get("system_name"),
            )
            for node in data.get("nodes", [])
        ]
        graph_edges = [
            GraphEdge(from_id=edge["from"], to_id=edge["to"], label=edge["label"])
            for edge in data.get("edges", [])
        ]

        return GraphQueryResponse(success=True, nodes=graph_nodes, edges=graph_edges, stats=data.get("stats"))
    except Exception as e:
        return GraphQueryResponse(success=False, nodes=[], edges=[], stats=None, message=str(e))


@router.get("/graph/node-counts", response_model=NodeCountsResponse)
def get_node_counts():
    try:
        data = get_node_counts_data()
        return NodeCountsResponse(success=data["success"], counts=data.get("counts"), message=data.get("message"))
    except Exception as e:
        return NodeCountsResponse(success=False, message=str(e))


@router.post("/material/preview", response_model=MaterialImportResponse)
async def preview_material(case_text: str = Form(""), files: Optional[List[UploadFile]] = File(None)):
    try:
        data = preview_material_data(case_text, files)
        return MaterialImportResponse(
            success=data.get("success", False),
            message=data.get("message"),
            result=data.get("result"),
        )
    except Exception as e:
        return MaterialImportResponse(success=False, message=str(e))


@router.post("/material/import-with-parsed", response_model=MaterialImportResponse)
async def import_material_with_parsed(parsed_data: str = Form(...), case_text: str = Form(""), files: Optional[List[UploadFile]] = File(None)):
    try:
        import json
        parsed_data_obj = json.loads(parsed_data)
        data = import_material_with_parsed_data(parsed_data_obj, case_text, files)
        return MaterialImportResponse(
            success=data.get("success", False),
            message=data.get("message"),
            result=data.get("result"),
        )
    except json.JSONDecodeError as e:
        return MaterialImportResponse(success=False, message=f"JSON格式错误: {str(e)}")
    except Exception as e:
        return MaterialImportResponse(success=False, message=str(e))


@router.post("/material/import", response_model=MaterialImportResponse)
async def import_material(case_text: str = Form(""), files: Optional[List[UploadFile]] = File(None)):
    try:
        data = import_material_data(case_text, files)
        return MaterialImportResponse(
            success=data.get("success", False),
            message=data.get("message"),
            result=data.get("result"),
        )
    except Exception as e:
        return MaterialImportResponse(success=False, message=str(e))
