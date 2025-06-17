import httpx
import filetype

from pathlib import Path
from openai._base_client import SyncAPIClient, AsyncAPIClient
from openai._resource import SyncAPIResource, AsyncAPIResource
from netmind.types import NetMindClient
from netmind.types.files import (
    FilePurpose, FilePresigned, BaseModel
)


class Files(SyncAPIResource):
    def create(
            self,
            file: Path | str,
            *,
            purpose: FilePurpose | str = FilePurpose.fine_tune,
    ):
        file_name = Path(file).name if isinstance(file, (Path, str)) else None
        assert file_name is not None, "File must be a path or string representing the file path."
        with open(file, 'rb') as f:
            mime = filetype.guess_mime(f)
            presign_url: FilePresigned = self._post(
                "/v1/files",
                body={
                    "file_name": file_name,
                    "purpose": purpose
                },
                cast_to=FilePresigned,
                options={"headers": {"file-content-type": mime}} if mime else {}
            )
            response = httpx.put(
                presign_url.presigned_url,
                content=f,
                headers={"Content-Type": mime} if mime else {}
            )
            response.raise_for_status()

        res: FilePresigned = self._get(
            f"/v1/files/{presign_url.id}/presigned_url",
            cast_to=FilePresigned,
        )
        res.id = presign_url.id
        return res


class AsyncFiles(AsyncAPIResource):
    import filetype
    from pathlib import Path
    import httpx

    async def create(
            self,
            file: Path | str,
            *,
            purpose: FilePurpose | str = FilePurpose.fine_tune,
    ):
        file_name = Path(file).name if isinstance(file, (Path, str)) else None
        assert file_name is not None, "File must be a path or string representing the file path."

        with open(file, 'rb') as f:
            file_bytes = f.read()
            mime = filetype.guess_mime(f)

        presign_url: FilePresigned = await self._post(
            "/v1/files",
            body={
                "file_name": file_name,
                "purpose": purpose
            },
            cast_to=FilePresigned,
            options={"headers": {"file-content-type": mime}} if mime else {}
        )

        async with httpx.AsyncClient() as client:
            response = await client.put(
                presign_url.presigned_url,
                content=file_bytes,
                headers={"Content-Type": mime} if mime else {}
            )
            response.raise_for_status()

        res: FilePresigned = await self._get(
            f"/v1/files/{presign_url.id}/presigned_url",
            cast_to=FilePresigned,
        )
        res.id = presign_url.id
        return res



