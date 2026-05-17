from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from passlib.exc import UnknownHashError
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..core.auth import hash_password, verify_password
from shared.models import User

if TYPE_CHECKING:
    from shared.models import User

logger = logging.getLogger(__name__)

ROLE_HIERARCHY: dict[str, int] = {
    "USER": 1,
    "ADMIN": 2,
    "SUPER_ADMIN": 3,
}


def normalize_role(role: str | None) -> str:
    normalized = str(role or "USER").strip().upper()
    if normalized not in ROLE_HIERARCHY:
        return "USER"
    return normalized


def role_rank(role: str | None) -> int:
    return ROLE_HIERARCHY.get(normalize_role(role), 0)


def get_user_by_email(session: Session, email: str) -> User | None:
    return session.scalar(select(User).where(User.email == email.lower().strip()))


def get_user_by_identifier(session: Session, identifier: str) -> User | None:
    normalized = str(identifier or "").strip().lower()
    if not normalized:
        return None
    if "@" in normalized:
        return get_user_by_email(session, normalized)
    statement = select(User).where(
        or_(
            func.lower(User.full_name) == normalized,
            func.lower(func.split_part(User.email, "@", 1)) == normalized,
        )
    )
    return session.scalar(statement)


def get_user_by_id(session: Session, user_id: int) -> User | None:
    return session.scalar(select(User).where(User.id == user_id))


def create_user(
    session: Session,
    email: str,
    full_name: str,
    password: str,
    *,
    role: str = "USER",
    is_active: bool = True,
) -> User:
    existing = get_user_by_email(session, email)
    if existing is not None:
        raise ValueError("Email already registered")

    user = User(
        email=email.lower().strip(),
        full_name=full_name.strip(),
        hashed_password=hash_password(password),
        role=normalize_role(role),
        is_active=bool(is_active),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def authenticate_user(session: Session, identifier: str, password: str) -> User | None:
    user = get_user_by_identifier(session, identifier)
    if user is None or not bool(user.is_active):
        return None
    try:
        if not verify_password(password, user.hashed_password):
            return None
    except UnknownHashError:
        logger.warning("Unsupported password hash format for user email=%s", user.email)
        return None
    user.last_login = datetime.now(timezone.utc)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def list_users(session: Session, *, actor: User | None = None) -> list[User]:
    statement = select(User).order_by(User.created_at.desc(), User.id.desc())
    if actor is not None:
        actor_rank = role_rank(actor.role)
        lower_roles = [r for r, rank in ROLE_HIERARCHY.items() if rank < actor_rank]
        statement = statement.where(User.role.in_(lower_roles), User.id != actor.id)
    return list(session.scalars(statement).all())


def update_user_access(session: Session, user: User, *, role: str | None = None, is_active: bool | None = None) -> User:
    if role is not None:
        user.role = normalize_role(role)
    if is_active is not None:
        user.is_active = bool(is_active)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def ensure_bootstrap_admin(session: Session, *, email: str, password: str, full_name: str) -> User | None:
    normalized_email = str(email or "").strip().lower()
    normalized_password = str(password or "").strip()
    if not normalized_email or not normalized_password:
        return None
    existing = get_user_by_email(session, normalized_email)
    if existing is not None:
        if str(existing.role or "").upper() != "SUPER_ADMIN":
            existing.role = "SUPER_ADMIN"
            existing.is_active = True
            session.add(existing)
            session.commit()
            session.refresh(existing)
        return existing
    admin = User(
        email=normalized_email,
        full_name=(full_name or "Administrator").strip() or "Administrator",
        hashed_password=hash_password(normalized_password),
        role="SUPER_ADMIN",
        is_active=True,
    )
    session.add(admin)
    session.commit()
    session.refresh(admin)
    return admin
