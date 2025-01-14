import axios from "axios";
import { getFilterParams, getSortParam, ontologyMap } from "./util";

// retrieve the full list of documents, with filters/sort
export const retrieveDocumentsList = async ({
    projectId,
    field,
    direction,
    filters,
}) => {
    let params = {};
    if (projectId) {
        params.project = projectId;
    }
    if (field && direction) {
        params.ordering = getSortParam({ field, direction });
    }
    if (filters) {
        params = { ...params, ...getFilterParams({ filters }) };
    }
    return await axios.get("/documents/", { params });
};

// retrieve single document by ID
export const retrieveDocument = async (documentId) =>
    await axios.get(`/documents/${documentId}/`);

// retrieve types list for a specific transcription on a document
export const retrieveTranscriptionOntology = async ({
    documentId,
    transcriptionId,
    category,
    sortField,
    sortDirection,
}) => {
    let params = {};
    if (sortField && sortDirection) {
        params.ordering = getSortParam({
            field: sortField,
            direction: sortDirection,
        });
    }
    return await axios.get(
        `/documents/${documentId}/transcriptions/${transcriptionId}/types/${ontologyMap[
            category
        ]}/`,
        { params },
    );
};

// retrieve characters, sorted by character or frequency, for a specific transcription on a document
export const retrieveTranscriptionCharacters = async ({
    documentId,
    transcriptionId,
    field,
    direction,
}) => {
    let params = {};
    if (field && direction) {
        params.ordering = getSortParam({ field, direction });
    }
    return await axios.get(
        `/documents/${documentId}/transcriptions/${transcriptionId}/characters/`,
        { params },
    );
};

// retrieve the total number of characters in a specific transcription level on a document
export const retrieveTranscriptionCharCount = async ({
    documentId,
    transcriptionId,
}) => {
    return await axios.get(
        `/documents/${documentId}/transcriptions/${transcriptionId}/character_count/`,
    );
};

// retrieve document parts
export const retrieveDocumentParts = async ({
    documentId,
    field,
    direction,
}) => {
    let params = {};
    if (field && direction) {
        params.ordering = getSortParam({ field, direction });
    }
    return await axios.get(`/documents/${documentId}/parts/`, { params });
};
// create a new document
export const createDocument = async ({
    name,
    project,
    mainScript,
    readDirection,
    linePosition,
    tags,
}) =>
    await axios.post("/documents/", {
        name,
        project,
        main_script: mainScript,
        read_direction: readDirection,
        line_offset: linePosition,
        tags,
    });

// delete a document
export const deleteDocument = async ({ documentId }) =>
    await axios.delete(`/documents/${documentId}/`);

// edit a document
export const editDocument = async (
    documentId,
    { name, project, mainScript, readDirection, linePosition, tags },
) =>
    await axios.put(`/documents/${documentId}/`, {
        name,
        project,
        main_script: mainScript,
        read_direction: readDirection,
        line_offset: linePosition,
        tags,
    });

// retrieve document metadata
export const retrieveDocumentMetadata = async (documentId) =>
    await axios.get(`/documents/${documentId}/metadata/`);

// create document metadata
export const createDocumentMetadata = async ({ documentId, metadatum }) =>
    await axios.post(`/documents/${documentId}/metadata/`, {
        ...metadatum,
    });

// update document metadata
export const updateDocumentMetadata = async ({ documentId, metadatum }) =>
    await axios.put(`/documents/${documentId}/metadata/${metadatum.pk}/`, {
        ...metadatum,
    });

// delete document metadata
export const deleteDocumentMetadata = async ({ documentId, metadatumId }) =>
    await axios.delete(`/documents/${documentId}/metadata/${metadatumId}/`);

// retrieve document models
export const retrieveDocumentModels = async (documentId) =>
    await axios.get("/models/", {
        params: {
            documents: documentId,
        },
    });

// share this document with a group or user
export const shareDocument = async ({ documentId, group, user }) =>
    await axios.post(`/documents/${documentId}/share/`, {
        group,
        user,
    });

// queue the segmentation task for this document
export const segmentDocument = async ({ documentId, override, model, steps }) =>
    await axios.post(`/documents/${documentId}/segment/`, {
        override,
        model,
        steps,
    });

// queue the transcription task for this document
export const transcribeDocument = async ({ documentId, model, layerName }) =>
    await axios.post(`/documents/${documentId}/transcribe/`, {
        model,
        name: layerName,
    });

// retrieve textual witnesses for use in alignment
export const retrieveTextualWitnesses = async () =>
    await axios.get("/textual-witnesses/");

// queue the alignment task for this document
export const alignDocument = async ({
    documentId,
    beamSize,
    existingWitness,
    fullDoc,
    gap,
    layerName,
    maxOffset,
    merge,
    ngram,
    regionTypes,
    threshold,
    transcription,
    witnessFile,
}) => {
    // need to use FormData to handle witness file upload
    const formData = new FormData();
    formData.append("beam_size", beamSize);
    formData.append("existing_witness", existingWitness);
    formData.append("full_doc", fullDoc);
    formData.append("gap", gap);
    formData.append("layer_name", layerName);
    formData.append("max_offset", maxOffset);
    formData.append("merge", merge);
    formData.append("n_gram", ngram);
    formData.append("region_types", regionTypes);
    formData.append("threshold", threshold);
    formData.append("transcription", transcription);
    formData.append("witness_file", witnessFile);
    const headers = { "Content-Type": "multipart/form-data" };
    return await axios.post(`/documents/${documentId}/align/`, formData, {
        headers,
    });
};

// queue the export task for this document
export const exportDocument = async ({
    documentId,
    fileFormat,
    includeImages,
    regionTypes,
    transcription,
}) =>
    await axios.post(`/documents/${documentId}/export/`, {
        region_types: regionTypes,
        file_format: fileFormat,
        include_images: includeImages,
        transcription,
    });

// queue the import task for this document
export const queueImport = async ({ documentId, params }) =>
    await axios.post(`/documents/${documentId}/import/`, params);

// retrieve latest tasks for a document
export const retrieveDocumentTasks = async ({ documentId }) =>
    await axios.get("/tasks/", {
        params: {
            document: documentId,
        },
    });

// cancel a task on a document by pk
export const cancelTask = async ({ documentId, taskReportId }) =>
    await axios.post(`/documents/${documentId}/cancel_tasks/`, {
        task_report: taskReportId,
    });

export const createTranscriptionLayer = async ({ documentId, layerName }) =>
    await axios.post(`/documents/${documentId}/transcriptions/`, {
        name: layerName,
    });
