from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ...core.dependencies import get_current_user, require_admin
from shared.models import User
from ...core.schemas import (
    AdminUserCreateRequest,
    AdminUserPatchRequest,
    UserListResponse,
    UserResponse,
)
from ...database import get_session
from ...services.user_service import (
    create_user,
    get_user_by_id,
    list_users,
    normalize_role,
    role_rank,
    update_user_access,
)

router = APIRouter(tags=["users"])


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def admin_create_user(
    payload: AdminUserCreateRequest,
    session: Session = Depends(get_session),
    _admin: User = Depends(require_admin),
) -> User:
    try:
        creator_rank = role_rank(_admin.role)
        requested_role = normalize_role(payload.role)
        requested_rank = role_rank(requested_role)
        if requested_rank >= creator_rank:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot create account with equal or higher role.")
        if requested_role == "ADMIN" and normalize_role(_admin.role) != "SUPER_ADMIN":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only SUPER_ADMIN can create ADMIN accounts.")
        return create_user(
            session,
            email=payload.email,
            full_name=payload.full_name,
            password=payload.password,
            role=requested_role,
            is_active=payload.is_active,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        session.close()


@router.get("", response_model=UserListResponse)
def admin_list_users(
    session: Session = Depends(get_session),
    _admin: User = Depends(require_admin),
) -> dict:
    try:
        items = list_users(session, actor=_admin)
        return {"count": len(items), "items": items}
    finally:
        session.close()


@router.patch("/{user_id}", response_model=UserResponse)
def admin_patch_user(
    user_id: int,
    payload: AdminUserPatchRequest,
    session: Session = Depends(get_session),
    _admin: User = Depends(require_admin),
) -> User:
    try:
        target = get_user_by_id(session, user_id)
        if target is None:
            raise HTTPException(status_code=404, detail="User not found")
        if payload.role is None and payload.is_active is None:
            return target
        if target.id == _admin.id and (payload.role is not None or payload.is_active is not None):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot change your own role or active status.")
        actor_role = normalize_role(_admin.role)
        actor_rank = role_rank(actor_role)
        target_role = normalize_role(target.role)
        target_rank = role_rank(target_role)
        if target_rank >= actor_rank:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot modify users with equal or higher role.")
        if payload.role is not None:
            next_role = normalize_role(payload.role)
            next_rank = role_rank(next_role)
            if next_rank >= actor_rank:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot assign equal or higher role.")
            if next_role == "ADMIN" and actor_role != "SUPER_ADMIN":
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only SUPER_ADMIN can modify ADMIN accounts.")
        if target_role == "ADMIN" and actor_role != "SUPER_ADMIN":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only SUPER_ADMIN can modify ADMIN accounts.")
        return update_user_access(session, target, role=payload.role, is_active=payload.is_active)
    finally:
        session.close()


@router.get("/{user_id}/queries")
def admin_get_user_queries(
    user_id: int,
    session: Session = Depends(get_session),
    _admin: User = Depends(require_admin),
) -> dict:
    from sqlalchemy import select as sa_select
    from shared.models import QueryHistory
    try:
        rows = session.scalars(
            sa_select(QueryHistory)
            .where(QueryHistory.user_id == user_id)
            .order_by(QueryHistory.created_at.desc())
        ).all()
        items = [
            {
                "query_id": r.query_id,
                "query_text": r.query_text,
                "query_image_url": r.query_image_url,
                "video_id": r.video_id,
                "storage_path": None,
                "created_at": r.created_at.isoformat(),
                "updated_at": r.updated_at.isoformat(),
            }
            for r in rows
        ]
        return {"count": len(items), "items": items}
    finally:
        session.close()
