import apiClient from './api';
import { User, UserCreateRequest } from '../types/user';

export const userService = {
  createUser: async (data: UserCreateRequest): Promise<User> => {
    const response = await apiClient.post<User>('/api/users', data);
    return response.data;
  },

  getUserById: async (id: number): Promise<User> => {
    const response = await apiClient.get<User>(`/api/users/${id}`);
    return response.data;
  },
};

export default userService;
