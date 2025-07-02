import os
import re
import httpx
import filetype

from pathlib import Path
from typing import List, Union
from openai._resource import SyncAPIResource, AsyncAPIResource
from netmind.types.files import (
    FilePurpose, FilePresigned,
    FileObject, FileId
)


def sanitize_filename(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    return f"{clean_name}{ext}"


class Files(SyncAPIResource):
    def create(
            self,
            file: Path | str,
            *,
            purpose: FilePurpose | str = FilePurpose.fine_tune,
    ) -> FileId:
        file_name = Path(file).name if isinstance(file, (Path, str)) else None
        assert file_name is not None, "File must be a path or string representing the file path."
        with open(file, 'rb') as f:
            mime = filetype.guess_mime(f)
            presign_url: FilePresigned = self._post(
                "/v1/files",
                body={
                    "file_name": sanitize_filename(file_name),
                    "purpose": purpose
                },
                cast_to=FilePresigned,
                options={"headers": {"file-content-type": mime}} if mime else {}
            )
            response = httpx.put(
                presign_url.presigned_url,
                content=f,
                headers={"Content-Type": mime} if mime else {},
                timeout=300
            )
            response.raise_for_status()
        return FileId(id=presign_url.id)

    def retrieve(self, file_id: str) -> FileObject:
        if not file_id:
            raise ValueError(f"Expected a non-empty value for `file_id` but received {file_id!r}")
        return self._get(
            f"/v1/files/{file_id}",
            cast_to=FileObject,
        )

    def list(self) -> List[FileObject]:
        return self._get(
            "/v1/files",
            cast_to=List[FileObject],
        )

    def delete(self, file_id: str) -> None:
        if not file_id:
            raise ValueError(f"Expected a non-empty value for `file_id` but received {file_id!r}")
        self._delete(f"/v1/files/{file_id}", cast_to=Union[None])

    def retrieve_url(self, file_id: str) -> FilePresigned:
        if not file_id:
            raise ValueError(f"Expected a non-empty value for `file_id` but received {file_id!r}")
        res: FilePresigned = self._get(
            f"/v1/files/{file_id}/presigned_url",
            cast_to=FilePresigned,
        )
        res.id = file_id
        return res


class AsyncFiles(AsyncAPIResource):
    async def create(
            self,
            file: Path | str,
            *,
            purpose: FilePurpose | str = FilePurpose.fine_tune,
    ) -> FileId:
        file_name = Path(file).name if isinstance(file, (Path, str)) else None
        assert file_name is not None, "File must be a path or string representing the file path."

        with open(file, 'rb') as f:
            file_bytes = f.read()
            mime = filetype.guess_mime(f)

        presign_url: FilePresigned = await self._post(
            "/v1/files",
            body={
                "file_name": sanitize_filename(file_name),
                "purpose": purpose
            },
            cast_to=FilePresigned,
            options={"headers": {"file-content-type": mime}} if mime else {}
        )

        async with httpx.AsyncClient() as client:
            response = await client.put(
                presign_url.presigned_url,
                content=file_bytes,
                headers={"Content-Type": mime} if mime else {},
                timeout=300
            )
            response.raise_for_status()
        return FileId(id=presign_url.id)

    async def retrieve(self, file_id: str) -> FileObject:
        if not file_id:
            raise ValueError(f"Expected a non-empty value for `file_id` but received {file_id!r}")
        return await self._get(
            f"/v1/files/{file_id}",
            cast_to=FileObject,
        )

    async def list(self) -> List[FileObject]:
        return await self._get(
            "/v1/files",
            cast_to=List[FileObject],
        )

    async def delete(self, file_id: str) -> None:
        if not file_id:
            raise ValueError(f"Expected a non-empty value for `file_id` but received {file_id!r}")
        await self._delete(f"/v1/files/{file_id}", cast_to=Union[None])

    async def retrieve_url(self, file_id: str) -> FilePresigned:
        if not file_id:
            raise ValueError(f"Expected a non-empty value for `file_id` but received {file_id!r}")
        res: FilePresigned = await self._get(
            f"/v1/files/{file_id}/presigned_url",
            cast_to=FilePresigned,
        )
        res.id = file_id
        return res
