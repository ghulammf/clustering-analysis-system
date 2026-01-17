import axios from "axios"
import API_BASE_URL from "../lib/axios"

const mainService = {
    uploads: async (formData: FormData) => {
        const response = await axios.post(`${API_BASE_URL}/api/upload`, formData)
        return response.data
    },
    getKmeans: async () => {
        const response = await axios.get(`${API_BASE_URL}/api/kmeans`)
        return response.data
    },
    getDbscan: async () => {
        const response = await axios.get(`${API_BASE_URL}/api/dbscan`)
        return response.data
    },
    getHierarchical: async () => {
        const response = await axios.get(`${API_BASE_URL}/api/hierarchical`)
        return response.data
    }
}

export default mainService